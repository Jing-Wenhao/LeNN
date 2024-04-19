from matplotlib import pyplot as plt
import numpy as np
from ase.db import connect

def plot_SHE(semiconductors):
    db = connect('H.db')
    for surface in semiconductors.keys():
        e_0 = 0
        g_0 = 0
        e = []
        for i in db.select():
            specie = i.get('species')
            terminal = i.get('terminal')
            # ads = atom.get('ads')
            ads_e_single = i.get('ads_e_single')
            spe_ter = specie+ '-' + str(terminal)
            if spe_ter == surface:
                    e.append(ads_e_single)
        RHE = np.linspace(0, 1.6, 500)
        RHE_real = []
        for rhe in RHE:
            RHE_real.append(round(-1 * rhe, 1))
        # lines = {}
        for j in range(len(e)):
            G = e[j] - j * RHE
            plt.plot(RHE, G, label=str(j) + 'H')
        plt.xticks(RHE[::100], RHE_real[::100])
        plt.axis([0, 1.5, -8, 4])
        plt.title('%s' % surface)
        plt.xlabel('RHE')
        plt.ylabel('ΔG/eV')
        plt.legend()
        plt.savefig(fname=r'curve\%s.png' % surface, dpi=600)
        plt.show()

        lines = {}

        for j in range(len(e)):
            G = e[j] - j * RHE
            lines['%d' % j] = G

        min_G = lines['0']
        min_G_lines = []
        for x_index in range(len(min_G)):
            min_G_lines.append(0)
        line_index = 0
        for line in lines.values():
            for x_index in range(len(line)):
                if line[x_index] < min_G[x_index]:
                    min_G[x_index] = line[x_index]
                    min_G_lines[x_index] = line_index
            line_index += 1
        # print(min_G)

        cross_RHE = []
        cross_line = []
        for i in range(len(min_G_lines) - 1):
            if min_G_lines[i] != min_G_lines[i + 1]:
                cross_RHE.append((RHE[i] + RHE[i + 1]) / 2)
                cross_line.append(min_G_lines[i])
                cross_line.append(min_G_lines[i + 1])
        cross_line = list(set(cross_line))
        cross_line.sort()
        # cross_RHE.insert(0, 0)
        print(cross_RHE)
        # print(cross_line)

        pH = np.linspace(0, 14, 50)
        count = 0
        SHE_y = {}
        bare = [0] * 50
        SHE_max = [-2.2] * 50
        color = ['#fcefee', '#fccde2', '#fc5c9c', '#c5e3f6']
        # color = ['#d7eaea', '#acdbdf', '#9692af', '#69779b']
        if len(cross_RHE) == 0:
            plt.plot(pH, bare, label='H-%d' % min_G_lines[0], color=color[0])
            plt.fill_between(pH, bare, -2.2, color=color[0])
        else:
            for l in range(len(cross_line)):
                if l == 0:
                    SHE_y[l] = bare
                    plt.plot(pH, SHE_y[l], label='H-%d' % cross_line[l], color=color[l])
                else:
                    if l != len(cross_line) - 1:
                        SHE = -cross_RHE[l - 1] - 0.0592 * pH
                        SHE_y[l] = SHE
                        plt.plot(pH, SHE_y[l], label='H-%d' % cross_line[l], color=color[l])
                        plt.fill_between(pH, SHE_y[l], SHE_y[l - 1], color=color[l - 1])
                    else:
                        SHE = -cross_RHE[l - 1] - 0.0592 * pH
                        SHE_y[l] = SHE
                        plt.plot(pH, SHE_y[l], label='H-%d' % cross_line[l], color=color[l])
                        plt.fill_between(pH, SHE_y[l], SHE_y[l - 1], color=color[l - 1])
                        plt.fill_between(pH, SHE_y[l], -2.2, color=color[l])
        # for l in range(len(cross_RHE)+2):
        #     if count == 0:
        #         SHE_y[l] = bare
        #         plt.plot(pH, SHE_y[l], label='bare')
        #     else:
        #         if count != len(cross_RHE) + 1:
        #             SHE = -cross_RHE[l-1] - 0.0592 * pH
        #             SHE_y[l] = SHE
        #             plt.plot(pH, SHE_y[l], label='H-%d' % cross_line[l-1])
        #             plt.fill_between(pH, SHE_y[l], SHE_y[l-1])
        #         else:
        #             SHE_y[l] = SHE_max
        #             plt.fill_between(pH, SHE_y[l], SHE_y[l-1])
        #     count += 1

        plt.plot(pH, -0.414 - 0.0592 * pH, color='black', linewidth=2, linestyle='dashed')
        plt.axis([0, 14, -2.2, 0])
        plt.legend()
        plt.xlabel('pH')
        plt.ylabel('SHE')
        plt.title('%s' % surface)
        plt.savefig(fname=r'SHE\%s.png' % surface, dpi=600)
        plt.show()