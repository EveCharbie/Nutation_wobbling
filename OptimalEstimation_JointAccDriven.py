

import biorbd_casadi as biorbd
import numpy as np
import ezc3d
import time
import casadi as cas
import pickle
import os
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    Solver,
    OdeSolver,
    Node,
    ConfigureProblem,
    DynamicsFunctions,
    ConstraintList,
    ConstraintFcn,
    BiorbdInterface,
)


def shift_by_2pi(biorbd_model, q, error_margin=0.35):
    n_q = biorbd_model.nbQ()
    q[4, :] = q[4, :] - ((2 * np.pi) * (np.mean(q[4, :]) / (2 * np.pi)).astype(int))
    for dof in range(6, n_q):
        q[dof, :] = q[dof, :] - ((2 * np.pi) * (np.mean(q[dof, :]) / (2 * np.pi)).astype(int))
        if ((2 * np.pi) * (1 - error_margin)) < np.mean(q[dof, :]) < ((2 * np.pi) * (1 + error_margin)):
            q[dof, :] = q[dof, :] - (2 * np.pi)
        elif ((2 * np.pi) * (1 - error_margin)) < -np.mean(q[dof, :]) < ((2 * np.pi) * (1 + error_margin)):
            q[dof, :] = q[dof, :] + (2 * np.pi)
    return q

def shift_by_pi(q, error_margin):
    if ((np.pi)*(1-error_margin)) < np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q - np.pi
    elif ((np.pi)*(1-error_margin)) < -np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q + np.pi
    return q

def reorder_markers(biorbd_model, c3d, frames, step_size=1, broken_dofs=None):
    markers = c3d['data']['points'][:3, :95, frames.start:frames.stop:step_size] / 1000
    c3d_labels = c3d['parameters']['POINT']['LABELS']['value'][:95]
    model_labels = [label.to_string() for label in biorbd_model.markerNames()]

    labels_index = []
    missing_markers_index = []
    for index_model, model_label in enumerate(model_labels):
        missing_markers_bool = True
        for index_c3d, c3d_label in enumerate(c3d_labels):
            if model_label in c3d_label:
                labels_index.append(index_c3d)
                missing_markers_bool = False
        if missing_markers_bool:
            labels_index.append(index_model)
            missing_markers_index.append(index_model)

    markers_reordered = np.zeros((3, markers.shape[1], markers.shape[2]))
    for index, label_index in enumerate(labels_index):
        if index in missing_markers_index:
            markers_reordered[:, index, :] = np.nan
        else:
            markers_reordered[:, index, :] = markers[:, label_index, :]

    model_segments = {
        'pelvis': {'markers': ['EIASD', 'CID', 'EIPSD', 'EIPSG', 'CIG', 'EIASG'], 'dofs': range(0, 6)},
        'thorax': {'markers': ['MANU', 'MIDSTERNUM', 'XIPHOIDE', 'C7', 'D3', 'D10'], 'dofs': range(6, 9)},
        'head': {'markers': ['ZYGD', 'TEMPD', 'GLABELLE', 'TEMPG', 'ZYGG'], 'dofs': range(9, 12)},
        'right_shoulder': {'markers': ['CLAV1D', 'CLAV2D', 'CLAV3D', 'ACRANTD', 'ACRPOSTD', 'SCAPD'], 'dofs': range(12, 14)},
        'right_arm': {'markers': ['DELTD', 'BICEPSD', 'TRICEPSD', 'EPICOND', 'EPITROD'], 'dofs': range(14, 17)},
        'right_forearm': {'markers': ['OLE1D', 'OLE2D', 'BRACHD', 'BRACHANTD', 'ABRAPOSTD', 'ABRASANTD', 'ULNAD', 'RADIUSD'], 'dofs': range(17, 19)},
        'right_hand': {'markers': ['METAC5D', 'METAC2D', 'MIDMETAC3D'], 'dofs': range(19, 21)},
        'left_shoulder': {'markers': ['CLAV1G', 'CLAV2G', 'CLAV3G', 'ACRANTG', 'ACRPOSTG', 'SCAPG'], 'dofs': range(21, 23)},
        'left_arm': {'markers': ['DELTG', 'BICEPSG', 'TRICEPSG', 'EPICONG', 'EPITROG'], 'dofs': range(23, 26)},
        'left_forearm': {'markers': ['OLE1G', 'OLE2G', 'BRACHG', 'BRACHANTG', 'ABRAPOSTG', 'ABRANTG', 'ULNAG', 'RADIUSG'], 'dofs': range(26, 28)},
        'left_hand': {'markers': ['METAC5G', 'METAC2G', 'MIDMETAC3G'], 'dofs': range(28, 30)},
        'right_thigh': {'markers': ['ISCHIO1D', 'TFLD', 'ISCHIO2D', 'CONDEXTD', 'CONDINTD'], 'dofs': range(30, 33)},
        'right_leg': {'markers': ['CRETED', 'JAMBLATD', 'TUBD', 'ACHILED', 'MALEXTD', 'MALINTD'], 'dofs': range(33, 34)},
        'right_foot': {'markers': ['CALCD', 'MIDMETA4D', 'MIDMETA1D', 'SCAPHOIDED', 'METAT5D', 'METAT1D'], 'dofs': range(34, 36)},
        'left_thigh': {'markers': ['ISCHIO1G', 'TFLG', 'ISCHIO2G', 'CONEXTG', 'CONDINTG'], 'dofs': range(36, 39)},
        'left_leg': {'markers': ['CRETEG', 'JAMBLATG', 'TUBG', 'ACHILLEG', 'MALEXTG', 'MALINTG', 'CALCG'], 'dofs': range(39, 40)},
        'left_foot': {'markers': ['MIDMETA4G', 'MIDMETA1G', 'SCAPHOIDEG', 'METAT5G', 'METAT1G'], 'dofs': range(40, 42)},
    }

    markers_idx_broken_dofs = []
    if broken_dofs is not None:
        for dof in broken_dofs:
            for segment in model_segments.values():
                if dof in segment['dofs']:
                    marker_positions = [index_model for marker_label in segment['markers'] for index_model, model_label in enumerate(model_labels) if marker_label in model_label]
                    if range(min(marker_positions), max(marker_positions) + 1) not in markers_idx_broken_dofs:
                        markers_idx_broken_dofs.append(range(min(marker_positions), max(marker_positions) + 1))

    return markers_reordered, markers_idx_broken_dofs

def adjust_number_shooting_points(number_shooting_points, frames):
    list_adjusted_number_shooting_points = []
    for frame_num in range(1, (abs(frames.stop - frames.start) - 1) // abs(frames.step) + 1):
        list_adjusted_number_shooting_points.append((abs(frames.stop - frames.start) - 1) // frame_num + 1)
    diff_shooting_points = [abs(number_shooting_points - point) for point in list_adjusted_number_shooting_points]
    step_size = diff_shooting_points.index(min(diff_shooting_points)) + 1
    adjusted_number_shooting_points = ((abs(frames.stop - frames.start) - 1) // step_size + 1) - 1

    return adjusted_number_shooting_points, step_size

def x_bounds(biorbd_model):
    pi = np.pi
    inf = 50000
    n_qdot = biorbd_model.nbQdot()

    qmin_base = [-3, -3, -1, -6*np.pi, -pi / 2.1, -8*np.pi]
    qmax_base = [3, 3, 7, 6*np.pi, pi / 2.1, 8*np.pi]
    qmin_thorax = [-pi / 2, -pi / 2.1, -pi / 2]
    qmax_thorax = [pi / 2, pi / 2.1, pi / 2]
    qmin_tete = [-pi / 2, -pi / 2.1, -pi / 2]
    qmax_tete = [pi / 2, pi / 2.1, pi / 2]
    qmin_epaule_droite = [-pi / 2, -pi / 2]
    qmax_epaule_droite = [pi / 2, pi / 2]
    qmin_bras_droit = [-pi, -pi / 2.1, -pi]
    qmax_bras_droit = [pi, pi / 2.1, pi]
    qmin_avantbras_droit = [-pi/8, -pi/2]
    qmax_avantbras_droit = [pi, pi]
    qmin_main_droite = [-pi / 2, -pi / 2]
    qmax_main_droite = [pi / 2, pi / 2]
    qmin_epaule_gauche = [-pi / 2, -pi / 2]
    qmax_epaule_gauche = [pi / 2, pi / 2]
    qmin_bras_gauche = [-pi, -pi / 2.1, -pi]
    qmax_bras_gauche = [pi, pi / 2.1, pi]
    qmin_avantbras_gauche = [-pi/8, -pi]
    qmax_avantbras_gauche = [pi, pi/2]
    qmin_main_gauche = [-3 * pi / 2, -3 * pi / 2]
    qmax_main_gauche = [3 * pi / 2, 3 * pi / 2]
    qmin_cuisse_droite = [-pi, -pi / 2.1, -pi / 2]
    qmax_cuisse_droite = [pi, pi / 2.1, pi / 2]
    qmin_jambe_droite = [-pi]
    qmax_jambe_droite = [pi/4]
    qmin_pied_droit = [-pi / 2, -pi / 2]
    qmax_pied_droit = [pi / 2, pi / 2]
    qmin_cuisse_gauche = [-pi, -pi / 2.1, -pi / 2]
    qmax_cuisse_gauche = [pi, pi / 2.1, pi / 2]
    qmin_jambe_gauche = [-pi]
    qmax_jambe_gauche = [pi/4]
    qmin_pied_gauche = [-pi / 2, -pi / 2]
    qmax_pied_gauche = [pi / 2, pi / 2]

    qdotmin_base = [-inf, -inf, -inf, -inf, -inf, -inf]
    qdotmax_base = [inf, inf, inf, inf, inf, inf]

    xmin = (qmin_base +  # q
            qmin_thorax +
            qmin_tete +
            qmin_epaule_droite +
            qmin_bras_droit +
            qmin_avantbras_droit +
            qmin_main_droite +
            qmin_epaule_gauche +
            qmin_bras_gauche +
            qmin_avantbras_gauche +
            qmin_main_gauche +
            qmin_cuisse_droite +
            qmin_jambe_droite +
            qmin_pied_droit +
            qmin_cuisse_gauche +
            qmin_jambe_gauche +
            qmin_pied_gauche +
            qdotmin_base +  # qdot
            [-200] * (n_qdot - 6))

    xmax = (qmax_base +
            qmax_thorax +
            qmax_tete +
            qmax_epaule_droite +
            qmax_bras_droit +
            qmax_avantbras_droit +
            qmax_main_droite +
            qmax_epaule_gauche +
            qmax_bras_gauche +
            qmax_avantbras_gauche +
            qmax_main_gauche +
            qmax_cuisse_droite +
            qmax_jambe_droite +
            qmax_pied_droit +
            qmax_cuisse_gauche +
            qmax_jambe_gauche +
            qmax_pied_gauche +
            qdotmax_base +  # qdot
            [200] * (n_qdot - 6))

    return xmin, xmax


def load_data_filename(subject, trial):
    if subject == 'DoCi':
        model_name = 'DoCi.s2mMod'
        # model_name = 'DoCi_SystemesDaxesGlobal_surBassin_rotAndre.s2mMod'
        if trial == '822':
            c3d_name = 'Do_822_contact_2.c3d'
            q_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(3089, 3360)
        if trial == '822_short':
            c3d_name = 'Do_822_contact_2_short.c3d'
            q_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(3119, 3330)
        if trial == '822_time_inverted':
            c3d_name = 'Do_822_contact_2_time_inverted.c3d'
            q_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(3089, 3360)
        elif trial == '44_1':
            c3d_name = 'Do_44_mvtPrep_1.c3d'
            q_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(2449, 2700)
        elif trial == '44_2':
            c3d_name = 'Do_44_mvtPrep_2.c3d'
            q_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(2599, 2850)
        elif trial == '44_3':
            c3d_name = 'Do_44_mvtPrep_3.c3d'
            q_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(4099, 4350)
    elif subject == 'JeCh':
        model_name = 'JeCh_201.s2mMod'
        # model_name = 'JeCh_SystemeDaxesGlobal_surBassin'
        if trial == '821_1':
            model_name = 'JeCh_201_bras_modifie.s2mMod'
            c3d_name = 'Je_821_821_1.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(2339, 2659)
        if trial == '821_821_1':
            model_name = 'JeCh_201_bras_modifie.s2mMod'
            c3d_name = 'Je_821_821_1.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(3129, 3419)
        if trial == '821_2':
            c3d_name = 'Je_821_821_2.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(2299, 2590)
        if trial == '821_3':
            c3d_name = 'Je_821_821_3.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(2299, 2590)
        if trial == '821_5':
            c3d_name = 'Je_821_821_5.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(2299, 2590)
        if trial == '833_1':
            c3d_name = 'Je_833_1.c3d'
            q_name = 'Je_833_1_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_1_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_1_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(1919, 2220)
            frames = range(2299, 2590)
        if trial == '833_2':
            c3d_name = 'Je_833_2.c3d'
            q_name = 'Je_833_2_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_2_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_2_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(1899, 2210)
            frames = range(2289, 2590)
        if trial == '833_3':
            c3d_name = 'Je_833_3.c3d'
            q_name = 'Je_833_3_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_3_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_3_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(2179, 2490)
            frames = range(2569, 2880)
        if trial == '833_4':
            c3d_name = 'Je_833_4.c3d'
            q_name = 'Je_833_4_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_4_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_4_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(2269, 2590)
            frames = range(2669, 2970)
        if trial == '833_5':
            c3d_name = 'Je_833_5.c3d'
            q_name = 'Je_833_5_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_5_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_5_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(2279, 2600)
            frames = range(2669, 2980)
    elif subject == 'BeLa':
        model_name = 'BeLa.s2mMod'
        # model_name = 'BeLa_SystemeDaxesGlobal_surBassin.s2mMod'
        if trial == '44_1':
            c3d_name = 'Ben_44_mvtPrep_1.c3d'
            q_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_Q.mat'
            qd_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_V.mat'
            qdd_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_A.mat'
            frames = range(1799, 2050)
        elif trial == '44_2':
            c3d_name = 'Ben_44_mvtPrep_2.c3d'
            q_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_Q.mat'
            qd_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_V.mat'
            qdd_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_A.mat'
            frames = range(2149, 2350)
        elif trial == '44_3':
            c3d_name = 'Ben_44_mvtPrep_3.c3d'
            q_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_Q.mat'
            qd_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_V.mat'
            qdd_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_A.mat'
            frames = range(2449, 2700)
    elif subject == 'GuSe':
        model_name = 'GuSe.s2mMod'
        # model_name = 'GuSe_SystemeDaxesGlobal_surBassin.s2mMod'
        if trial == '44_2':
            c3d_name = 'Gui_44_mvt_Prep_2.c3d'
            q_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_Q.mat'
            qd_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_V.mat'
            qdd_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_A.mat'
            frames = range(1649, 1850)
        elif trial == '44_3':
            c3d_name = 'Gui_44_mvt_Prep_3.c3d'
            q_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_Q.mat'
            qd_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_V.mat'
            qdd_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_A.mat'
            frames = range(1699, 1950)
        elif trial == '44_4':
            c3d_name = 'Gui_44_mvtPrep_4.c3d'
            q_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_Q.mat'
            qd_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_V.mat'
            qdd_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_A.mat'
            frames = range(1599, 1850)
    elif subject == 'SaMi':
        model_name = 'SaMi.s2mMod'
        if trial == '821_822_2':
            c3d_name = 'Sa_821_822_2.c3d'
            q_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(2909, 3220)
            frames = range(3299, 3590)
            # frames = range(3659, 3950)
        elif trial == '821_822_3':
            c3d_name = 'Sa_821_822_3.c3d'
            q_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3139, 3440)
        # elif trial == '821_822_4':
        #     c3d_name = 'Sa_821_822_4.c3d'
        #     q_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_Q.mat'
        #     qd_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_V.mat'
        #     qdd_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_A.mat'
        #     # frames = range(3509, 3820)
        #     frames = range(3909, 4190)
        # elif trial == '821_822_5':
        #     c3d_name = 'Sa_821_822_5.c3d'
        #     q_name = 'Sa_821_822_5_MOD200.00_GenderF_SaMig_Q.mat'
        #     qd_name = 'Sa_821_822_5_MOD200.00_GenderF_SaMig_V.mat'
        #     qdd_name = 'Sa_821_822_5_MOD200.00_GenderF_SaMig_A.mat'
        #     frames = range(3339, 3630)
        elif trial == '821_contact_1':
            c3d_name = 'Sa_821_contact_1.c3d'
            q_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3019, 3330)
        elif trial == '821_contact_2':
            c3d_name = 'Sa_821_contact_2.c3d'
            q_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3569, 3880)
        elif trial == '821_contact_3':
            c3d_name = 'Sa_821_contact_3.c3d'
            q_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3309, 3620)
        elif trial == '822_contact_1':
            c3d_name = 'Sa_822_contact_1.c3d'
            q_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(5009, 5310)
        elif trial == '821_seul_1':
            c3d_name = 'Sa_821_seul_1.c3d'
            q_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3349, 3650)
        elif trial == '821_seul_2':
            c3d_name = 'Sa_821_seul_2.c3d'
            q_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3429, 3740)
        elif trial == '821_seul_3':
            c3d_name = 'Sa_821_seul_3.c3d'
            q_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3209, 3520)
        elif trial == '821_seul_4':
            c3d_name = 'Sa_821_seul_4.c3d'
            q_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3309, 3620)
        elif trial == '821_seul_5':
            c3d_name = 'Sa_821_seul_5.c3d'
            q_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(2689, 3000)
        elif trial == 'bras_volant_1':
            c3d_name = 'Sa_bras_volant_1.c3d'
            q_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(0, 4657)
            # frames = range(649, 3950)
            # frames = range(649, 1150)
            # frames = range(1249, 1950)
            # frames = range(2549, 3100)
            frames = range(3349, 3950)
        elif trial == 'bras_volant_2':
            c3d_name = 'Sa_bras_volant_2.c3d'
            q_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(0, 3907)
            # frames = range(0, 3100)
            # frames = range(49, 849)
            # frames = range(1599, 2200)
            frames = range(2249, 3100)
    else:
        raise Exception(subject + ' is not a valid subject')

    data_filename = {
        'model': model_name,
        'c3d': c3d_name,
        'q': q_name,
        'qd': qd_name,
        'qdd': qdd_name,
        'frames': frames,
    }

    return data_filename


def rotating_gravity(biorbd_model, value):
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which decribed by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    gravity = biorbd_model.getGravity()
    gravity.applyRT(
        biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d()))
    biorbd_model.setGravity(gravity)


def root_explicit_dynamic(states, controls, parameters, nlp,):
    DynamicsFunctions.apply_parameters(parameters, nlp)
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    nb_root = nlp.model.nbRoot()
    qddot_joints = DynamicsFunctions.get(nlp.controls["qddot_joint"], controls)
    mass_matrix_nl_effects = nlp.model.InverseDynamics(q, qdot, cas.vertcat(cas.MX.zeros((nb_root, 1)), qddot_joints)).to_mx()[:6]
    mass_matrix = nlp.model.massMatrix(q).to_mx()
    mass_matrix_nl_effects_func = cas.Function("mass_matrix_nl_effects_func", [q, qdot, qddot_joints], [mass_matrix_nl_effects[:nb_root]]).expand()
    M_66 = mass_matrix[:nb_root, :nb_root]
    M_66_func = cas.Function("M66_func", [q], [M_66]).expand()
    qddot_root = cas.solve(M_66_func(q), -mass_matrix_nl_effects_func(q, qdot, qddot_joints), "ldl")
    return qdot, cas.vertcat(qddot_root, qddot_joints)


def custom_configure_root_explicit(ocp, nlp):
    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    configure_qddot_joint(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, root_explicit_dynamic, expand=False)


def configure_qddot_joint(nlp, as_states, as_controls):
    nb_root = nlp.model.nbRoot()
    name_qddot_joint = [str(i + nb_root) for i in range(nlp.model.nbQddot() - nb_root)]
    ConfigureProblem.configure_new_variable("qddot_joint", name_qddot_joint, nlp, as_states, as_controls)

def dynamics_root(m, X, Qddot_J):
    Q = X[:m.nbQ()]
    Qdot = X[m.nbQ():]
    Qddot = np.hstack((np.zeros((6,)), Qddot_J)) #qddot2
    NLEffects = m.InverseDynamics(Q, Qdot, Qddot).to_array()
    mass_matrix = m.massMatrix(Q).to_array()
    Qddot_R = np.linalg.solve(mass_matrix[:6, :6], -NLEffects[:6])
    Xdot = np.hstack((Qdot, Qddot_R, Qddot_J))
    return Xdot


def custom_func_track_markers(all_pn, target, slack):
    markers = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])

    return markers


def prepare_ocp(
        biorbd_model_path,
        final_time,
        number_shooting_points,
        markers_ref,
        q_init,
        qdot_init,
        qddot_init,
        xmin,
        xmax,
        min_torque_diff=False):


    model = biorbd.Model(biorbd_model_path)
    model.setGravity(biorbd.Vector3d(0, 0, -9.80639))
    biorbd_model = (model)

    qddot_min, qddot_max = -1000, 1000
    n_q = biorbd_model.nbQ()
    n_qddot = n_q - biorbd_model.nbRoot()

    objective_functions = ObjectiveList()
    # Tracking term
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS, node=Node.START, weight=1000, target=markers_ref)
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS, node=Node.END, weight=1000, target=markers_ref)
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, node=Node.ALL, weight=1, target=markers_ref)

    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, node=Node.ALL, weight=1e-7, key="q", target=q_init, multi_thread=False)
    # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, node=Node.ALL, weight=1e-5, key="qdot", target=qdot_init, multi_thread=False)
    # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, node=Node.ALL_SHOOTING, weight=1e-5, key="qddot_joint", target=qddot_init[6:, :-1], multi_thread=False)

    # Regularization terms
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joint", weight=1e-7)
    # if min_torque_diff:
    #     objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joint", derivative=True, weight=1e-5)
    # Extra regularization terms
    # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, node=Node.ALL, weight=1e-5, target=state_ref)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, node=Node.ALL, weight=1e-5, index=range(6, n_q))
    # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key="tau", weight=1e-7, target=tau_init)

    # Constraints
    constraints = ConstraintList()
    # constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, min_bound=markers_ref[:, 0]-0.1, max_bound=markers_ref[:, 0]+0.1)
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, min_bound=-0.01, max_bound=0.01, target=markers_ref[:, 0])
    # constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.MID, min_bound=markers_ref[:, 0]-1, max_bound=markers_ref[:, 0]+1)
    # for i in range(biorbd_model.nbMarkers()):
    #     constraints.add(custom_func_track_markers, node=Node.ALL, min_bound=markers_ref-0.5, max_bound=markers_ref+0.5)
    # MID et vérrifier que c'est le bon indice

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure_root_explicit, dynamic_function=root_explicit_dynamic)

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(min_bound=xmin, max_bound=xmax)

    # Initial guess
    X_init = InitialGuessList()
    X_init.add(np.concatenate([q_init, qdot_init]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(min_bound=[qddot_min] * n_qddot, max_bound=[qddot_max] * n_qddot)

    U_init = InitialGuessList()
    # Option to set initial guess to zero if it is abnormal
    # tau_init = np.zeros(tau_init.shape)
    U_init.add(qddot_init[6:, :-1], interpolation=InterpolationType.EACH_FRAME)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        ode_solver=OdeSolver.RK4(n_integration_steps=4),
        n_threads=4,
    )


if __name__ == "__main__":
    start = time.time()
    subject = 'SaMi'
    number_shooting_points = 100
    trial = '822_contact_1'
    print('Subject: ', subject, ', Trial: ', trial)

    # Choose between track_q or EKF for an initial guess
    initial_guess_track_Q = False

    # Choose to use the optimized gravity of track_q_gravity, which also sets the initial guess as track_q_gravity
    use_optimized_gravity = False
    if use_optimized_gravity:
        initial_guess_track_Q = False

    # Add trial to this list if adding a regularization term on the control derivative is judged necessary
    trial_needing_min_torque_diff = {
                                     # 'DoCi': ['822'],
                                     }

    min_torque_diff = False
    if subject in trial_needing_min_torque_diff.keys():
        if trial in trial_needing_min_torque_diff[subject]:
            min_torque_diff = True

    data_path = 'data/' + subject + '/'

    data_filename = load_data_filename(subject, trial)
    model_name = data_filename['model']
    c3d_name = data_filename['c3d']
    frames = data_filename['frames']

    biorbd_model_path = data_path + model_name
    biorbd_model = biorbd.Model(biorbd_model_path)
    c3d = ezc3d.c3d(data_path + c3d_name)

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)
    print('Adjusted number of shooting points: ', adjusted_number_shooting_points)
    print('Node step size: ', step_size)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    load_path = f'data/{subject}/EKF/'
    load_name = load_path + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_name, 'rb') as handle:
        EKF = pickle.load(handle)
    q_ref = shift_by_2pi(biorbd_model, EKF['q'][:, ::step_size])
    qdot_ref = EKF['qd'][:, ::step_size]
    qddot_ref = EKF['qdd'][:, ::step_size]

    xmin, xmax = x_bounds(biorbd_model)

    # Organize the markers in the same order as in the model
    markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames, step_size)

    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        markers_ref=markers_reordered, q_init=q_ref, qdot_init=qdot_ref, qddot_init=qddot_ref,
        xmin=xmin, xmax=xmax, min_torque_diff=min_torque_diff,
    )

    # --- Solve the program --- #
    ocp.add_plot_penalty()
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(3000)
    # solver.set_linear_solver("ma57")
    # solver.set_tol(1e-4)
    # solver.set_constr_viol_tol(1e-2)
    sol = ocp.solve(solver)
    stop = time.time()
    print('Runtime: ', stop - start)

    print('Number of shooting points: ', adjusted_number_shooting_points)

    # --- Get the results --- #
    q_sol = sol.states['q']
    qdot_sol = sol.states['qdot']
    qddot_sol = sol.controls['qddot_joint']

    # --- Save --- #
    save_path = 'data/optimizations/'
    save_name = save_path + os.path.splitext(c3d_name)[0] + "_N" + str(adjusted_number_shooting_points)

    get_gravity = cas.Function('get_gravity', [], [biorbd_model.getGravity().to_mx()], [], ['gravity'])
    gravity = get_gravity()['gravity'].full().squeeze()

    save_variables_name = save_name + ".pkl"
    with open(save_variables_name, 'wb') as handle:
        pickle.dump({'mocap': markers_reordered, 'duration': duration, 'frames': frames, 'step_size': step_size,
                     'q': q_sol, 'qdot': qdot_sol, 'qddot_joint': qddot_sol, 'gravity': gravity},
                    handle, protocol=3)

    # sol.graphs()

    import bioviz
    b = bioviz.Viz(biorbd_model_path)
    b.load_experimental_markers(markers_reordered)
    b.load_movement(q_sol)
    b.exec()

