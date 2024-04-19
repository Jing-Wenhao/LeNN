import numpy as np
from numpy import *
from ase.io import read
from hads import element_information

class Features_coord:
    def __init__(
            self,
            structure,
            cut = 1.05,
            max_coord = 4
    ):
        self.structure = structure
        self.id = id
        self.distances = structure.get_all_distances()
        self.z = structure.get_atomic_numbers()
        self.num_h = np.bincount(self.z)[1]
        self.symbols = structure.get_chemical_symbols()
        self.coord = structure.get_positions()
        self.fraction_coords = structure.get_scaled_positions()
        self.lattice = structure.get_cell()
        self.index_1 = -1
        self.cut = cut
        self.index_2 = self.find_second_atom()
        self.index_3 = self.find_third_atom(self.index_2)
        self.index_4 = self.find_forth_atom(self.index_3)
        self.max_coord = max_coord
        self.formula = structure.get_chemical_formula()
        self.e_all = self.calculate_valence(self.symbols)

    def calculate_valence(self, symbols):
        o = symbols.count('O')
        h = symbols.count('H')
        m = len(symbols) - o - h
        # forward = []
        # back = []
        # for i in formula:
        #     if i in element_information.back:
        #         back.append(i)
        #     else:
        #         forward.append(i)
        # length = len(forward)
        # valence_b = 0
        # valence_f = {}
        # for j in back:
        #     valence_b += element_information.data[j]['valence'] * formula[j]
        # if length == 1:
        #     valence_f[forward[0]] = int(valence_b * (-1) / formula[forward[0]])
        #     valence_b = int(valence_b / formula[back[0]])
        #     return valence_f, valence_b
        # if length == 2:
        #     valence_f[forward[0]] = input("Please enter the element %s's valence" % forward[0])
        #     valence_f[forward[1]] = input("Please enter the element %s's valence" % forward[1])
        #     return valence_b, valence_f
        return o * 2 / m

    def cal_distance(self, coordinate, target):
        return linalg.norm(target - coordinate, axis=1)

    def get_true_distances(self, index, r_judge=0.4, return_c=False):
        index = index
        coordinate = self.fraction_coords[index]
        a = delete(self.lattice[0], 2)
        b = delete(self.lattice[1], 2)
        x = coordinate[0]
        y = coordinate[1]
        a_len = linalg.norm(a)
        b_len = linalg.norm(b)
        fraction_1 = self.fraction_coords.copy()
        fraction_2 = self.fraction_coords.copy()
        fraction_3 = self.fraction_coords.copy()
        if (x > r_judge) and (x < (1 - r_judge)) and (y > r_judge) and (y < (1 - r_judge)):
            if return_c:
                all_coord = []
                all_coord.append(self.coord)
                all_coord.append(dot(fraction_1, self.lattice))
                all_coord.append(dot(fraction_2, self.lattice))
                all_coord.append(dot(fraction_3, self.lattice))
                return np.zeros(len(self.symbols), dtype=int), all_coord
            else:
                return self.distances[index]

        else:
            if (x <= r_judge):
                fraction_1[:, 0] = fraction_1[:, 0] - 1
                # print(fraction_1)
                if (y <= r_judge):
                    fraction_2[:, 1] = fraction_2[:, 1] - 1
                    fraction_3[:, 0:2] = fraction_3[:, 0:2] - 1
                elif (y >= (1 - r_judge)):
                    fraction_2[:, 1] = fraction_2[:, 1] + 1
                    fraction_3[:, 0] = fraction_3[:, 0] - 1
                    fraction_3[:, 1] = fraction_3[:, 1] + 1
                else:
                    pass
            elif (x >= (1 - r_judge)):
                fraction_1[:, 0] = fraction_1[:, 0] + 1
                if (y <= r_judge):
                    fraction_2[:, 1] = fraction_2[:, 1] - 1
                    fraction_3[:, 0] = fraction_3[:, 0] + 1
                    fraction_3[:, 1] = fraction_3[:, 1] - 1
                elif (y >= (1 - r_judge)):
                    fraction_2[:, 1] = fraction_2[:, 1] + 1
                    fraction_3[:, 0:2] = fraction_3[:, 0:2] + 1
                else:
                    pass
            else:
                if (y <= r_judge):
                    fraction_2[:, 1] = fraction_2[:, 1] - 1
                else:
                    fraction_2[:, 1] = fraction_2[:, 1] + 1
            # coordinates_1 = dot(fraction_1, lattice)
            # coordinates_2 = dot(fraction_2, lattice)
            # coordinates_3 = dot(fraction_3, lattice)
            all_coord = []
            all_coord.append(self.coord)
            all_coord.append(dot(fraction_1, self.lattice))
            all_coord.append(dot(fraction_2, self.lattice))
            all_coord.append(dot(fraction_3, self.lattice))
            # all_coord = list(set(all_coord))
            all_distances = []
            count = []
            for i in range(len(all_coord)):
                all_distances.append(self.cal_distance(self.coord[index], all_coord[i]))
                count.append(i)
            if not return_c:
                return np.min(np.array(all_distances), axis=0)
            else:
                return np.argmin(all_distances, axis=0), all_coord

    def get_symbols_only(self, index_first, index_last):
        if type(index_first) == int:
            index_first_l = []
            index_first_l.append(index_first)
            index_first = index_first_l
        else:
            index_first = list(index_first)
            index_last = list(index_last)
        symbols_first, symbols_last, index_last_temp = [],[],[]
        for i in index_first:
            symbols_first.append(self.symbols[i])
        for j in index_last:
            symbols_last.append(self.symbols[j])
        for index, s in enumerate(symbols_last):
            if s not in symbols_first:
                index_last_temp.append(index_last[index])
        return index_last_temp

    def find_second_atom(self):
        min = 100
        min_index = []
        # print(self.get_true_distances(index = -1))
        for index,d in enumerate(self.get_true_distances(index = -1)):
            if d <= min and d!=0 and index < len(self.distances) - self.num_h:
                min_index = index
                min = d

        return min_index

    def find_third_atom(self, index_2):

        distance = self.get_true_distances(index_2)
        symbol_2 = self.symbols[index_2]
        index_3 = []
        r2 = element_information.data[symbol_2]['r']
        for index, d in enumerate(distance):
            r1 = element_information.data[self.symbols[index]]['r']
            if d <= (r1 + r2) * self.cut and d != 0 and index < len(distance) - self.num_h:
                index_3.append(index)
        return self.get_symbols_only(index_2, index_3)

    def find_forth_atom(self, index_3):
        index_4 = []
        index_4_temp = []
        for i in index_3:
            index_4_temp.append(self.find_third_atom(i))
        # index_4_temp = set([item for sublist in index_4_temp for item in sublist])
        # for j in index_4_temp:
        #     if j not in index_3:
        #         index_4.append(j)
        return index_4_temp

    def ball_coord_new(self, index_2, index_3, index_4):

        # R: 吸附位点到配位原子质心的距离
        # d: 吸附位点到配位原子的距离向量
        # r: 配位原子质心到配位原子间的距离
        # e: 每个配位原子给质心传递的电子数，与d同维度
        # d_coord: 配位原子之间的距离向量，与d同维度
        # r_coord: 配位原子的原子半径（物理属性）

        where_min, all_coord = self.get_true_distances(index_2, return_c=True)
        coord_real = self.coord
        symbol = self.symbols[index_3[0]]
        r_coord = np.array(element_information.data[symbol]['r'])
        for i, c in enumerate(where_min):
            coord_real[i] = all_coord[c][i]

        if len(index_3) == 1:
            center = self.coord[index_3]
            R = self.get_true_distances(index_2)[index_3]
            r = 0
            d_coord = np.array(0)
            d = np.array(R)
            sort_d = np.argsort(d)
            # theta = np.array(0)

        if len(index_3) == 2:
            where_min, all_coord = self.get_true_distances(index_2, return_c=True)
            coord_real = self.coord
            for i, c in enumerate(where_min):
                coord_real[i] = all_coord[c][i]
            coord1 = coord_real[index_3[0]]
            coord2 = coord_real[index_3[1]]

            center = np.sum(coord_real[index_3], axis=0) / 2
            R = np.linalg.norm(center - coord_real[index_2])
            r = np.linalg.norm(center - coord_real[index_3][0])
            d = np.linalg.norm(coord_real[index_2] - coord_real[index_3], axis=1)
            d_coord = np.array([np.linalg.norm(coord1 - coord2), np.linalg.norm(coord1 - coord2)])
            sort_d = np.argsort(d)
            # theta = np.array([0, 0.5 * np.pi])

        if len(index_3) == 3:
            d = self.get_true_distances(index_2)[index_3]
            sort_d = np.argsort(d)

            coord1 = coord_real[index_3[sort_d[0]]]
            coord2 = coord_real[index_3[sort_d[1]]]
            coord3 = coord_real[index_3[sort_d[2]]]

            center = self.round_center(coord1, coord2, coord3)
            R = np.linalg.norm(center - self.coord[index_2])

            d_coord = []
            d_coord.append(np.linalg.norm(coord1 - coord2))
            d_coord.append(np.linalg.norm(coord2 - coord3))
            d_coord.append(np.linalg.norm(coord1 - coord3))
            d_coord = np.array(d_coord)
            r = np.linalg.norm(center - self.coord[index_3], axis=1)[0]

            # rela_coord1 = coord1 - center
            # rela_coord2 = coord2 - center
            # rela_coord3 = coord3 - center
            # d1 = np.linalg.norm(rela_coord1 - rela_coord2)
            # d2 = np.linalg.norm(rela_coord1 - rela_coord3)
            # theta1 = np.arcsin(d1/2 / r)
            # theta2 = np.arcsin(d2/2 / r)
            # theta = np.array([0, theta1, theta2])

        if len(index_3) == 4:
            d = self.get_true_distances(index_2)[index_3]
            sort_d = np.argsort(d)

            coord1 = coord_real[index_3[sort_d[0]]]
            coord2 = coord_real[index_3[sort_d[1]]]
            coord3 = coord_real[index_3[sort_d[2]]]
            coord4 = coord_real[index_3[sort_d[3]]]

            a = np.linalg.norm(coord1 - coord2)
            b = np.linalg.norm(coord1 - coord3)
            c = np.linalg.norm(coord1 - coord4)
            d = np.linalg.norm(coord2 - coord3)
            e = np.linalg.norm(coord2 - coord4)
            f = np.linalg.norm(coord3 - coord4)

            alpha1 = []
            alpha1.append(np.arccos((a**2 + b**2 - d**2) / (2 * a * b)))
            alpha1.append(np.arccos((a**2 + c**2 - e**2) / (2 * a * c)))
            alpha1.append(np.arccos((b**2 + c**2 - f**2) / (2 * b * c)))
            max_alpha = 0
            max_alpha_index = 0
            for i, a in enumerate(alpha1):
                if a > max_alpha:
                    max_alpha_index = i
            if i == 0:
                d_coord = np.array([a, e, f, b])
            elif i == 1:
                d_coord = np.array([a, d, f, c])
            elif i == 2:
                d_coord = np.array([b, d, e, c])


            center = self.ball_center(coord1, coord2, coord3, coord4)
            R = np.linalg.norm(center - self.coord[index_2])
            d = self.get_true_distances(index_2)[index_3]
            r = np.linalg.norm(center - coord_real[index_3], axis=1)[0]
            # d_coord = []
            # d_coord.append(np.linalg.norm(coord1 - coord2))
            # d_coord.append(np.linalg.norm(coord1 - coord3))
            # d_coord.append(np.linalg.norm(coord1 - coord4))
            # d_coord.append(np.linalg.norm(coord2 - coord3))
            # d_coord.append(np.linalg.norm(coord2 - coord4))
            # d_coord.append(np.linalg.norm(coord3 - coord4))
            # d_coord = np.array(d_coord)
            # rela_coord1 = coord1 - center
            # rela_coord2 = coord2 - center
            # rela_coord3 = coord3 - center
            # rela_coord4 = coord4 - center
            # m = np.matrix([rela_coord1, rela_coord2, rela_coord3, rela_coord4])
            # theta = np.array(arccos(m[:,2] / np.matrix(np.linalg.norm(m, axis=1)).T)).ravel()
            # theta = theta - theta[0]
            # fai = np.array(arctan(m[:, 1] / m[:, 0])).ravel()
            # fai = fai - fai[0]
            # d1 = np.linalg.norm(rela_coord1 - rela_coord2)
            # d2 = np.linalg.norm(rela_coord1 - rela_coord3)
            # theta1 = np.arcsin(d1 / 2 / r)
            # theta2 = np.arcsin(d2 / 2 / r)
            # theta = np.array([0, theta1, theta2])

        e = []
        for i,j in enumerate(index_3):
            # where_min, all_coord = self.get_true_distances(j, return_c=True)
            # coord_real_3 = self.coord
            # for a, c in enumerate(where_min):
            #     coord_real_3[a] = all_coord[c][a]
            coord_O = index_4[i]
            dis_O = self.get_true_distances(j)[coord_O]
            for k,l in enumerate(coord_O):
                if l == index_2:
                    tag = k
            b = 1/dis_O[tag] / np.sum(1/dis_O)
            e.append(self.e_all * b)
        e = np.array(e)
        if len(index_3) != 1:
            e = e[sort_d]
            d = np.sort(d)

        h = self.count_H_4(index_3, index_4)

        X = np.zeros((4,4))
        X[0, :len(index_3)] = d
        X[1, :len(index_3)] = d_coord
        X[2, :len(index_3)] = e
        X[3, 0] = R
        X[3, 1] = r
        X[3, 2] = r_coord
        X[3, 3] = h
        X = X.ravel()
        X = np.append(X, len(index_3))
        return X.ravel()



    def round_center(self, coord1, coord2, coord3):
        x1, y1, z1 = coord1[0], coord1[1], coord1[2]
        x2, y2, z2 = coord2[0], coord2[1], coord2[2]
        x3, y3, z3 = coord3[0], coord3[1], coord3[2]
        A1 = y1*z2 - y1*z3 - z1*y2 + z1*y3 + y2*z3 -y3*z2
        B1 = -x1*z2 + x1*z3 + z1*x2 - z1*x3 - x2*z3 +x3*z2
        C1 = x1*y2 - x1*y3 - y1*x2 + y1*x3 + x2*y3 - x3*y2
        D1 = -x1*y2*z3 + x1*y3*z2 + x2*y1*z3 - x3*y1*z2 - x2*y3*z1 + x3*y2*z1
        A2 = 2*(x2-x1)
        B2 = 2*(y2-y1)
        C2 = 2*(z2-z1)
        D2 = x1**2 + y1**2 + z1**2 - (x2**2 + y2**2 + z2**2)
        A3 = 2 * (x3 - x1)
        B3 = 2 * (y3 - y1)
        C3 = 2 * (z3 - z1)
        D3 = x1 ** 2 + y1 ** 2 + z1 ** 2 - (x3 ** 2 + y3 ** 2 + z3 ** 2)
        transfer_1 = np.array([[A1,B1,C1], [A2,B2,C2], [A3,B3,C3]])
        transfer_2 = np.array([D1,D2,D3])
        center = -np.dot(np.linalg.inv(transfer_1), transfer_2)

        return center

    def ball_center(self, coord1, coord2, coord3, coord4):
        x1, y1, z1 = coord1[0], coord1[1], coord1[2]
        x2, y2, z2 = coord2[0], coord2[1], coord2[2]
        x3, y3, z3 = coord3[0], coord3[1], coord3[2]
        x4, y4, z4 = coord4[0], coord4[1], coord4[2]

        a1, b1, c1 = x1-x2, y1-y2, z1-z2
        a2, b2, c2 = x3-x4, y3-y4, z3-z4
        a3, b3, c3 = x2-x3, y2-y3, z2-z3

        P = 0.5 * (x1**2 - x2**2 + y1**2 - x2**2 + z1**2 - z2**2)
        Q = 0.5 * (x3**2 - x4**2 + y3**2 - x4**2 + z3**2 - z4**2)
        R = 0.5 * (x2**2 - x3**2 + y2**2 - x3**2 + z2**2 - z3**2)

        D = a1*b2*c3 + b1*c2*a3 + c1*a2*b3 - c1*b2*a3 - b1*a2*b3 - a1*c2*b3
        Dx = P*b2*c3 + b1*c2*R + c1*Q*b3 - c1*b2*R - b1*Q*c3 - P*c2*b3
        Dy = a1*Q*c3 + P*c2*a3 + c1*a2*R - c1*Q*a3 - P*a2*c3 - a1*c2*R
        Dz = a1*b2*R + b1*Q*a3 + P*a2*b3 - P*b2*a3 - b1*a2*R - a1*Q*b3

        center = np.array([Dx/D, Dy/D, Dz/D])

        return center

    def ball_coord(self, index_2, index_3, index_4):
        coord_center = self.coord[index_2]
        coord_3 = self.coord[index_3]
        coord_ball = zeros(3 * self.max_coord).reshape((3, self.max_coord))
        r = np.max(self.get_true_distances(index_2)[index_3])
        # R = np.zeros(4)
        for i,j in enumerate(index_3):
            if self.get_true_distances(index_2)[j] == r:
                index_round = j
                index_max = i


        where_min, all_coord = self.get_true_distances(index_2, return_c=True)
        coord_real = self.coord     #以index_2(即与H直接相连的O原子)为核心的真实距离计算
        for i, c in enumerate(where_min):
            coord_real[i] = all_coord[c][i]

        rela_coord = self.get_rela_coord(coord_real, index_2, index_3, index_round)

        m = np.matrix(rela_coord)
        R = np.array(linalg.norm(rela_coord, axis=1))

        theta = np.array(arccos(m[:,2] / np.matrix(np.linalg.norm(m, axis=1)).T)).ravel()
        theta = theta - theta[index_max]
        fai = np.array(arctan(m[:,1] / m[:,0])).ravel()
        fai = fai - fai[index_max]
        coord_ball[0, :len(index_3)] = R
        coord_ball[1, :len(index_3)] = theta
        coord_ball[2, :len(index_3)] = fai

        return coord_ball

    def ball_coord_o(self, index_2, index_3, index_4):
        coord_center = self.coord[index_2]

    def get_rela_coord(self, coord, index_2, index_3, index_round):
        rela_coord = []
        for i in index_3:
            rela_coord.append(coord[i] - coord[index_2])
        return rela_coord

    def count_H_4(self, index_3, index_4):
        H_c = 0
        H_ensemble = []
        count = 0
        for i in index_3:
            count_H = 0
            for j in index_4[count]:
                min_temp = 100
                min_index = 0
                for index, d in enumerate(self.structure.get_all_distances()[j]):
                    if d != 0 and d<=min_temp:
                        min_index = index
                        min_temp = d
                # where_h = argmin(self.structure.get_all_distances()[j])
                if self.symbols[min_index] == 'H' and min_index not in H_ensemble:
                    H_ensemble.append(min_index)
                    count_H += 1
            count += 1
            H_c += count_H

        return H_c


    def get_coord_properties(self, index_3, index_4):
        # transfer_mat = np.matrix(np.zeros((4,2)))
        # transfer_mat[:,0] = 1
        coord_properties = zeros(2 * self.max_coord).reshape((2, self.max_coord))
        symbol = self.symbols[index_3[0]]
        e = []
        O_c = []
        H_c = []
        # Z = element_information.data[symbol]['number'] * 4
        Z = []
        count = 0
        for i in index_3:
            O_c.append(len(index_4[count]))
            # print(len(index_4[count]))
            count_H = 0
            for j in index_4[count]:
                if self.symbols[argmin(self.structure.get_all_distances)] == 'H':
                    count_H += 1
            count += 1
            H_c.append(count_H)
        # while len(H_c) < 4:
        #     H_c.append(0)
        # transfer_mat = np.matrix(H_c)
        # R = dot(R.T, transfer_mat)

        O_c = np.matrix(O_c)
        # print(O_c.shape)
        # print(coord_properties)
        coord_properties[0, :len(index_3)] = O_c
        coord_properties[1, :len(index_3)] = H_c
        coord_properties = list(coord_properties.ravel())
        coord_properties.append(element_information.data[symbol]['number'])
        coord_properties.append(element_information.data[symbol]['electronegativity'])
        coord_properties.append(element_information.data[symbol]['OutermostElectron'])
        coord_properties.append(element_information.data[symbol]['r'])
        # coord_properties[2, :4] = Z
        # Z = np.array(Z)
        # print(Z)
        # print(coord_properties)
        return np.array(coord_properties)


    def get_features(self):
        atom_1 = self.structure[-1]
        id = []
        distance_first = self.distances[-1]
        index_2 = self.find_second_atom()
        if self.judge():
            id.append(self.id)
            index_3 = self.find_third_atom(index_2)
            index_4 = self.find_forth_atom(index_3)
            # print(index_4)
            # r = self.ball_coord(index_2, index_3, index_4)
            X = self.ball_coord_new(index_2, index_3, index_4)
            p = self.get_coord_properties(index_3, index_4)
            X = concatenate((X, p), axis=0).ravel()
            return X
            # return concatenate((R, d, r, d_coord, e, r_coord), axis=0).ravel()
        # else:
        #     return self.ball_coord_o(index_2, index_3)


    def judge(self):
        if len(self.index_3) != 0:
            if self.symbols[self.index_2] != 'O' :
                return False
            else:
                return 2
        else:
            return False


        # avg_coord = average(m, axis=0).reshape((3,1))
        # R[0:len(index_3)] = np.array(linalg.norm(rela_coord,  axis=1))
        # print(R.shape)
        # O_c = []
        # H_c = []
        # # Z = element_information.data[symbol]['number'] * 4
        # Z = []
        # count = 0
        # for i in index_3:
        #     O_c.append(len(index_4[count]))
        #     # print(len(index_4[count]))
        #     count_H = 0
        #     for j in index_4[count]:
        #         if self.symbols[argmin(self.structure.get_all_distances)] == 'H':
        #             count_H += 1
        #     count += 1
        #     H_c.append(count_H)
        # while len(H_c) < 4:
        #     H_c.append(0)
        # while len(O_c) < 4:
        #     O_c.append(0)
        # H_c = np.array(H_c)
        # O_c = np.array(O_c)
        # transfer_mat = np.matrix((O_c, H_c))
        #
        # R_1 = np.array(dot(transfer_mat, R).ravel()) / len(index_3)
        # R_1 = R_1.reshape((2,))

        # R = np.concatenate((R, R_1), axis=0)

        # symbol = self.symbols[index_3[0]]
        # r_2 = np.array(linalg.norm(avg_coord))
        # theta = np.array(arccos(avg_coord[2] / np.matrix(np.linalg.norm(avg_coord)))).ravel()
        # fai = np.array(arctan(avg_coord[1] / avg_coord[0])).ravel()
        # R = np.insert(R, R.shape[0], r_2)
        # R = np.insert(R, R.shape[0], theta)
        # R = np.insert(R, R.shape[0], fai)
        # R = np.insert(R, R.shape[0],element_information.data[symbol]['number'])