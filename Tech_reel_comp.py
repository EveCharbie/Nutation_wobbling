
import pickle
import numpy as np
import biorbd
import matplotlib.pyplot as plt
import bioviz
import ezc3d


def dynamics_root(m, X, Qddot_J):
    Q = X[:m.nbQ()]
    Qdot = X[m.nbQ():]
    Qddot = np.hstack((np.zeros((6,)), Qddot_J))
    NLEffects = m.InverseDynamics(Q, Qdot, Qddot).to_array()
    mass_matrix = m.massMatrix(Q).to_array()
    Qddot_R = np.linalg.solve(mass_matrix[:6, :6], -NLEffects[:6])
    Xdot = np.hstack((Qdot, Qddot_R, Qddot_J))
    return Xdot

def runge_kutta_4(m, x0, Qddot_J, t, N, n_step):
    h = t / (N-1) / n_step
    x = np.zeros((x0.shape[0], n_step + 1))
    root_acc = np.zeros((6, n_step + 1))
    x[:, 0] = x0
    for i in range(1, n_step + 1):
        k1 = dynamics_root(m, x[:, i - 1], Qddot_J)
        k2 = dynamics_root(m, x[:, i - 1] + h / 2 * k1, Qddot_J)
        k3 = dynamics_root(m, x[:, i - 1] + h / 2 * k2, Qddot_J)
        k4 = dynamics_root(m, x[:, i - 1] + h * k3, Qddot_J)
        x[:, i] = np.hstack((x[:, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)))
        root_acc[:, i] = h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)[m.nbQ():m.nbQ()+6]
    return x, root_acc

def integrate_plots(m, X0, Q, Qdot, Qddot_joints, n_step, N, t):
    fig, axs = plt.subplots(1, 3)
    axs = np.ravel(axs)
    X_tous = X0
    U_root_tous = np.zeros((6, ))
    for i in range(N - 1):
        X, U_root = runge_kutta_4(m, X0, Qddot_joints[:, i], t[-1], N, n_step)
        plot_index = 0
        for k in range(3, 6):
            axs[plot_index].plot(np.arange(i * n_step, (i + 1) * n_step + 1), np.reshape(X[k, :], n_step + 1), ':',
                        label=f'{m.nameDof()[k].to_string()}')
            plot_index += 1
        X_tous = np.vstack((X_tous, np.hstack((X[:6, -1], Q[6:, i+1], X[m.nbQ():m.nbQ()+6, -1], Qdot[6:, i+1]))))
        U_root_tous = np.vstack((U_root_tous, U_root[:, -1]))
        X0 = X[:, -1]

    fig.suptitle('Reintegration')
    plt.show()
    return X_tous, U_root_tous


if __name__ == "__main__":

    PLOT_FLAG = False

    model_path = "data/SaMi/SaMi.s2mMod"
    model = biorbd.Model(model_path)

    ### test OK :)
    # N_test = 4
    # Q_test = np.zeros((model.nbQ(), N_test))
    # Q_test[3, 1] = np.pi/2
    # Q_test[4, 2] = np.pi/2
    # Q_test[5, 3] = np.pi/2
    #
    # bodyInertia = np.zeros((N_test, 3, 3))
    # pelvis_JCS = np.zeros((N_test, 3, 3))
    # localInertia = np.zeros((N_test, 3, 3))
    # bodyAngularVelocity = np.zeros((N_test, 3))
    # for i in range(N_test):
    #     bodyInertia[i, :, :] = model.bodyInertia(Q_test[:, i]).to_array()
    #     pelvis_JCS[i, :, :] = model.globalJCS(Q_test[:, i], 0).to_array()[:3, :3]
    #     localInertia[i, :, :] = pelvis_JCS[i, :, :].T @ bodyInertia[i, :, :] @ pelvis_JCS[i, :, :]


    with open(r"data/optimizations/Sa_822_contact_1_N100.pkl", "rb") as input_file:
        data = pickle.load(input_file)

    Q = data["q"]
    Qdot = data["qdot"]
    Qddot_J = data["qddot_joint"]
    N = np.shape(data["q"])[1]
    t = np.linspace(0, data["duration"], N)
    AngulatPosition = Q[3:6, :]
    AngulatVelocity = Qdot[3:6, :]



    bodyInertia = np.zeros((N, 3, 3))
    pelvis_JCS = np.zeros((N, 3, 3))
    localInertia = np.zeros((N, 3, 3))
    bodyAngularVelocity = np.zeros((N, 3))
    for i in range(N):
        bodyInertia[i, :, :] = model.bodyInertia(Q[:, i]).to_array()
        pelvis_JCS[i, :, :] = model.globalJCS(Q[:, i], 0).to_array()[:3, :3]
        localInertia[i, :, :] = pelvis_JCS[i, :, :].T @ bodyInertia[i, :, :] @ pelvis_JCS[i, :, :]
        bodyAngularVelocity[i, :] = model.bodyAngularVelocity(Q[:, i], Qdot[:, i]).to_array()


    if PLOT_FLAG:
        label_1 = ["Somersault", "Tilt", "Twist"]
        fig, axs = plt.subplots(3, 1)
        for i in range(3):
            axs[i].plot(AngulatPosition[i, :], '-r', label=("Q rotation" if i == 2 else None))
            axs[i].plot(AngulatVelocity[i, :], '-b', label=("Qdot rotation" if i == 2 else None))
            axs[i].plot(bodyAngularVelocity[:, i], '-k', label=("body velocity" if i == 2 else None))
            axs[i].set_title(label_1[i])
        plt.legend()

        fig, ax = plt.subplots(3, 3)
        axs = ax.ravel()
        for i in range(3):
            for j in range(3):
                axs[i*3+j].plot(bodyInertia[:, i, j], '-b')
                axs[i*3+j].set_ylim(np.min(bodyInertia)-0.1, np.max(bodyInertia)+0.1)
                axs[i*3+j].plot(np.array([0, N]), np.zeros((2, 1)), '-k')
        plt.suptitle("Body inertia matrix")

        fig, ax = plt.subplots(3, 3)
        axs = ax.ravel()
        for i in range(3):
            for j in range(3):
                axs[i*3+j].plot(localInertia[:, i, j], '-r')
                axs[i*3+j].set_ylim(np.min(localInertia)-0.1, np.max(localInertia)+0.1)
                axs[i*3+j].plot(np.array([0, N]), np.zeros((2, 1)), '-k')
        plt.suptitle("Inertia matrix (frame f)")
        plt.show()


    # --------------------------------------------------------------------------


    # b = bioviz.Viz(model_path)
    # b.load_movement(Q)
    # b.exec()


    n_step = 5

    m_SaMi = biorbd.Model(model_path)

    X0 = np.hstack((Q[:, 0], Qdot[:, 0]))
    X_tous, U_root = integrate_plots(m_SaMi, X0, Q, Qdot, Qddot_J, n_step, N, t)
    labels = ["Somersault", "Tilt", "Twist"]
    fig, axs = plt.subplots(1, 3)
    axs = np.ravel(axs)
    for i in range(3):
        axs[i].plot(X_tous[3+i, :], '-r', label=labels[i])
        axs[i].plot(X_tous[m_SaMi.nbQ()+3+i, :], '-g', label=labels[i]+" velocity")
        axs[i].plot(U_root[:, 3+i], '-b', label=labels[i] + " acceleration")
    plt.suptitle("With initial rotation velocity")
    plt.legend()

    b = bioviz.Viz(model_path)
    b.load_movement(X_tous[:, :m_SaMi.nbQ()].T)
    b.exec()



    X0 = np.hstack((Q[:, 0], np.zeros((6,)), Qdot[6:, 0]))
    m_SaMi.setGravity(np.array((0, 0, 0)))
    X_tous_without, U_root_without = integrate_plots(m_SaMi, X0, Q, Qdot, Qddot_J, n_step, N, t)
    labels = ["Somersault", "Tilt", "Twist"]
    fig, axs = plt.subplots(1, 3)
    axs = np.ravel(axs)
    for i in range(3):
        axs[i].plot(X_tous_without[3+i, :], '-r', label=labels[i])
        axs[i].plot(X_tous_without[m_SaMi.nbQ()+3+i, :], '-g', label=labels[i]+" velocity")
        axs[i].plot(U_root_without[3+i, :], '-b', label=labels[i] + " acceleration")
    plt.suptitle("Without initial rotation velocity")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(X_tous[4, :], '-r', label='With')
    plt.plot(X_tous_without[4, :], '-b', label='Without')
    plt.plot(X_tous[4, :] - X_tous_without[4, :], '-k', label='With - without')
    plt.legend()
    plt.show()

    b = bioviz.Viz(model_path)
    b.load_movement(X_tous_without[:, :m_SaMi.nbQ()].T)
    b.exec()













