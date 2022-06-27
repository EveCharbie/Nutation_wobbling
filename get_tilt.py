
import numpy as np
import matplotlib.pyplots as plt
import biorbd
import bioviz
import pickle


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
    u = np.zeros((x0.shape[0], n_step + 1))
    x[:, 0] = x0
    for i in range(1, n_step + 1):
        k1 = dynamics_root(m, x[:, i - 1], Qddot_J)
        k2 = dynamics_root(m, x[:, i - 1] + h / 2 * k1, Qddot_J)
        k3 = dynamics_root(m, x[:, i - 1] + h / 2 * k2, Qddot_J)
        k4 = dynamics_root(m, x[:, i - 1] + h * k3, Qddot_J)
        x[:, i] = np.hstack((x[:, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)))
        u[:, i] = h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)[m.nbQ():m.nbQ()+6]
    return x, u

def integrate_plots(m, Q, Qdot, Qddot_joints, n_step, N, t):
    fig, axs = plt.subplots(1, 3)
    axs = np.ravel(axs)
    X0 = np.vstack((Q[:, 0], Qdot[:, 0]))
    X_tous = X0
    U_root = np.zeros((6, N))
    for i in range(N - 1):
        X, U_root[:, i] = runge_kutta_4(m, X0, Qddot_joints[i, :], t[-1], N, n_step)
        plot_index = 0
        for k in range(3, 6):
            axs[plot_index].plot(np.arange(i * n_step, (i + 1) * n_step + 1), np.reshape(X[k, :], n_step + 1), ':',
                        label=f'{m.nameDof()[k].to_string()}')
            plot_index += 1
        X_tous = np.vstack((X_tous, X[:6, -1], Q[6:, i+1], X[m.nbQ():m.nbQ()+6, -1], Qdot[6:, i+1]))
        X0 = X[:, -1]

    fig.suptitle('Reintegration')
    plt.show()
    return X_tous, U_root

def integrate_plots_withoutInitialRotations(m, Q, Qdot, Qddot_joints, n_step, N, t):
    fig, axs = plt.subplots(1, 3)
    axs = np.ravel(axs)
    X0 = np.vstack((Q[:, 0], np.zeros((6, 1)), Qdot[6:, 0]))
    X_tous = X0
    U_root = np.zeros((6, N))
    for i in range(N - 1):
        X, U_root[:, i] = runge_kutta_4(m, X0, Qddot_joints[i, :], t[-1], N, n_step)
        plot_index = 0
        for k in range(3, 6):
            axs[plot_index].plot(np.arange(i * n_step, (i + 1) * n_step + 1), np.reshape(X[k, :], n_step + 1), ':',
                        label=f'{m.nameDof()[k].to_string()}')
            plot_index += 1
        X_tous = np.vstack((X_tous, X[:6, -1], Q[6:, i+1], X[m.nbQ():m.nbQ()+6, -1], Qdot[6:, i+1]))
        X0 = X[:, -1]

    fig.suptitle('Reintegration without initial velocity')
    plt.show()
    return X_tous, U_root



###################################################################################
N = 100
n_step = 5

model_path_SaMi = "/home/user/Documents/Programmation/Eve/AnthropoImpactOnTech/Models/SaMi.bioMod"  #######################
m_SaMi = biorbd.Model(model_path_SaMi)

with open(r"822_contact_2.pkl", "rb") as input_file: ######################################################################
    data = pickle.load(input_file)

Q = data["states"]["q"]
Qdot = data["states"]["q_dot"]
Qddot_joints = data["controls"]["q_ddot_joints"]
t = np.linspace(0, data["parameters"]["t"], N)
N = np.shape(Q)[1]

# m_SaMi.setGravity(np.array((0, 0, 0)))

X_tous, U_root = integrate_plots(m_SaMi, Q, Qdot, Qddot_joints, n_step, N, t)
labels = ["Somersault", "Tilt", "Twist"]
fig, axs = plt.subplots(1, 3)
axs = np.ravel(axs)
for i in range(3):
    axs[i].plot(X_tous[3+i, :], '-r', label=labels[i])
    axs[i].plot(X_tous[m_SaMi.nbQ()+3+i, :], '-g', label=labels[i]+" velocity")
    axs[i].plot(U_root[3+i, :], '-b', label=labels[i] + " acceleration")
plt.suptitle("With initial rotation velocity")


X_tous_without, U_root_without = integrate_plots_withoutInitialRotations(m_SaMi, Q, Qdot, Qddot_joints, n_step, N, t)
labels = ["Somersault", "Tilt", "Twist"]
fig, axs = plt.subplots(1, 3)
axs = np.ravel(axs)
for i in range(3):
    axs[i].plot(X_tous_without[3+i, :], '-r', label=labels[i])
    axs[i].plot(X_tous_without[m_SaMi.nbQ()+3+i, :], '-g', label=labels[i]+" velocity")
    axs[i].plot(U_root_without[3+i, :], '-b', label=labels[i] + " acceleration")
plt.suptitle("Without initial rotation velocity")
plt.show()

plt.figure()
plt.plot(X_tous[4, :], '-r', label='With')
plt.plot(X_tous_without[4, :], '-b', label='Without')
plt.plot(X_tous[4, :] - X_tous_without[4, :], '-k', label='With - without')
plt.show()



b = bioviz.Viz(model_path_JeCh)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()








