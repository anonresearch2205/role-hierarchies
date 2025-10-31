import matplotlib.pyplot as plt
import numpy as np

data = {
    "fw1": {"MinRolesRH_Roles": 64, "MinRolesRH_Edges": 2211, "NewRolesRH_Roles": 129, "NewRolesRH_Edges": 1956},
    "fw2": {"MinRolesRH_Roles": 10, "MinRolesRH_Edges": 1023, "NewRolesRH_Roles": 19, "NewRolesRH_Edges": 977},
    "hc": {"MinRolesRH_Roles": 14, "MinRolesRH_Edges": 198, "NewRolesRH_Roles": 35, "NewRolesRH_Edges": 376},
    "domino": {"MinRolesRH_Roles": 20, "MinRolesRH_Edges": 722, "NewRolesRH_Roles": 43, "NewRolesRH_Edges": 739},
    "apj": {"MinRolesRH_Roles": 453, "MinRolesRH_Edges": 4066, "NewRolesRH_Roles": 684, "NewRolesRH_Edges": 4170},
    "as": {"MinRolesRH_Roles": 178, "MinRolesRH_Edges": 9910, "NewRolesRH_Roles": 338, "NewRolesRH_Edges": 9098},
    "al": {"MinRolesRH_Roles": 398, "MinRolesRH_Edges": 88355, "NewRolesRH_Roles": 546, "NewRolesRH_Edges": 92220},
    "emea": {"MinRolesRH_Roles": 34, "MinRolesRH_Edges": 7280, "NewRolesRH_Roles": 34, "NewRolesRH_Edges": 7246},
    "univ": {"MinRolesRH_Roles": 18, "MinRolesRH_Edges": 682, "NewRolesRH_Roles": 46, "NewRolesRH_Edges": 646},
    "mailer": {"MinRolesRH_Roles": 565, "MinRolesRH_Edges": 5357, "NewRolesRH_Roles": 1047, "NewRolesRH_Edges": 5365},
    "small 01": {"MinRolesRH_Roles": 24, "MinRolesRH_Edges": 254, "NewRolesRH_Roles": 49, "NewRolesRH_Edges": 277},
    "small 02": {"MinRolesRH_Roles": 25, "MinRolesRH_Edges": 378, "NewRolesRH_Roles": 51, "NewRolesRH_Edges": 398},
    "small 03": {"MinRolesRH_Roles": 25, "MinRolesRH_Edges": 406, "NewRolesRH_Roles": 51, "NewRolesRH_Edges": 439},
    "small 04": {"MinRolesRH_Roles": 25, "MinRolesRH_Edges": 486, "NewRolesRH_Roles": 50, "NewRolesRH_Edges": 505},
    "small 05": {"MinRolesRH_Roles": 49, "MinRolesRH_Edges": 530, "NewRolesRH_Roles": 100, "NewRolesRH_Edges": 578},
    "small 06": {"MinRolesRH_Roles": 50, "MinRolesRH_Edges": 722, "NewRolesRH_Roles": 105, "NewRolesRH_Edges": 764},
    "small 07": {"MinRolesRH_Roles": 34, "MinRolesRH_Edges": 1246, "NewRolesRH_Roles": 60, "NewRolesRH_Edges": 1273},
    "small 08": {"MinRolesRH_Roles": 50, "MinRolesRH_Edges": 1011, "NewRolesRH_Roles": 93, "NewRolesRH_Edges": 1041},
    "medium 01": {"MinRolesRH_Roles": 150, "MinRolesRH_Edges": 3118, "NewRolesRH_Roles": 305, "NewRolesRH_Edges": 3266},
    "medium 02": {"MinRolesRH_Roles": 150, "MinRolesRH_Edges": 5217, "NewRolesRH_Roles": 234, "NewRolesRH_Edges": 5287},
    "medium 03": {"MinRolesRH_Roles": 199, "MinRolesRH_Edges": 5931, "NewRolesRH_Roles": 306, "NewRolesRH_Edges": 6005},
    "medium 04": {"MinRolesRH_Roles": 200, "MinRolesRH_Edges": 4451, "NewRolesRH_Roles": 428, "NewRolesRH_Edges": 4657},
    "medium 05": {"MinRolesRH_Roles": 200, "MinRolesRH_Edges": 6516, "NewRolesRH_Roles": 458, "NewRolesRH_Edges": 6745},
    "medium 06": {"MinRolesRH_Roles": 250, "MinRolesRH_Edges": 7542, "NewRolesRH_Roles": 401, "NewRolesRH_Edges": 7678},
    "large 01": {"MinRolesRH_Roles": 250, "MinRolesRH_Edges": 8677, "NewRolesRH_Roles": 464, "NewRolesRH_Edges": 8862},
    "large 02": {"MinRolesRH_Roles": 500, "MinRolesRH_Edges": 10112, "NewRolesRH_Roles": 1080, "NewRolesRH_Edges": 10662},
    "large 03": {"MinRolesRH_Roles": 499, "MinRolesRH_Edges": 7560, "NewRolesRH_Roles": 1126, "NewRolesRH_Edges": 8053},
    "large 04": {"MinRolesRH_Roles": 400, "MinRolesRH_Edges": 11003, "NewRolesRH_Roles": 883, "NewRolesRH_Edges": 11458},
    "large 05": {"MinRolesRH_Roles": 400, "MinRolesRH_Edges": 15985, "NewRolesRH_Roles": 670, "NewRolesRH_Edges": 16235},
    "large 06": {"MinRolesRH_Roles": 500, "MinRolesRH_Edges": 11260, "NewRolesRH_Roles": 1105, "NewRolesRH_Edges": 11856},
    "comp 01.1": {"MinRolesRH_Roles": 400, "MinRolesRH_Edges": 9521, "NewRolesRH_Roles": 759, "NewRolesRH_Edges": 9811},
    "comp 01.2": {"MinRolesRH_Roles": 400, "MinRolesRH_Edges": 10849, "NewRolesRH_Roles": 737, "NewRolesRH_Edges": 11129},
    "comp 01.3": {"MinRolesRH_Roles": 400, "MinRolesRH_Edges": 10079, "NewRolesRH_Roles": 733, "NewRolesRH_Edges": 10355},
    "comp 01.4": {"MinRolesRH_Roles": 400, "MinRolesRH_Edges": 11407, "NewRolesRH_Roles": 786, "NewRolesRH_Edges": 11727},
    "comp 02.1": {"MinRolesRH_Roles": 2036, "MinRolesRH_Edges": 62122, "NewRolesRH_Roles": 7527, "NewRolesRH_Edges": 162859},
    "comp 02.2": {"MinRolesRH_Roles": 2109, "MinRolesRH_Edges": 89623, "NewRolesRH_Roles": 7257, "NewRolesRH_Edges": 200399},
    "comp 02.3": {"MinRolesRH_Roles": 2132, "MinRolesRH_Edges": 115045, "NewRolesRH_Roles": 7191, "NewRolesRH_Edges": 251442},
    "comp 02.4": {"MinRolesRH_Roles": 5013, "MinRolesRH_Edges": 457733, "NewRolesRH_Roles": 0, "NewRolesRH_Edges": 0},
    "comp 03.1": {"MinRolesRH_Roles": 5483, "MinRolesRH_Edges": 353314, "NewRolesRH_Roles": 19708, "NewRolesRH_Edges": 464780},
    "comp 03.2": {"MinRolesRH_Roles": 10033, "MinRolesRH_Edges": 702826, "NewRolesRH_Roles": 10523, "NewRolesRH_Edges": 702913},
    "comp 03.3": {"MinRolesRH_Roles": 5600, "MinRolesRH_Edges": 543482, "NewRolesRH_Roles": 5600, "NewRolesRH_Edges": 543482},
    "comp 03.4": {"MinRolesRH_Roles": 10022, "MinRolesRH_Edges": 1155174, "NewRolesRH_Roles": 10854, "NewRolesRH_Edges": 1178587},
    "comp 04.1": {"MinRolesRH_Roles": 3594, "MinRolesRH_Edges": 116417, "NewRolesRH_Roles": 11865, "NewRolesRH_Edges": 315139},
    "comp 04.2": {"MinRolesRH_Roles": 3602, "MinRolesRH_Edges": 137766, "NewRolesRH_Roles": 11637, "NewRolesRH_Edges": 353387},
    "comp 04.3": {"MinRolesRH_Roles": 3625, "MinRolesRH_Edges": 146073, "NewRolesRH_Roles": 12237, "NewRolesRH_Edges": 393560},
    "comp 04.4": {"MinRolesRH_Roles": 3644, "MinRolesRH_Edges": 138562, "NewRolesRH_Roles": 0, "NewRolesRH_Edges": 0},
    "rw 01": {"MinRolesRH_Roles": 463, "MinRolesRH_Edges": 354814, "NewRolesRH_Roles": 1835, "NewRolesRH_Edges": 384587},
    "2level 01": {"MinRolesRH_Roles": 15, "MinRolesRH_Edges": 407, "NewRolesRH_Roles": 45, "NewRolesRH_Edges": 473},
    "2level 02": {"MinRolesRH_Roles": 15, "MinRolesRH_Edges": 430, "NewRolesRH_Roles": 32, "NewRolesRH_Edges": 455},
    "2level 03": {"MinRolesRH_Roles": 25, "MinRolesRH_Edges": 489, "NewRolesRH_Roles": 49, "NewRolesRH_Edges": 509},
    "2level 04": {"MinRolesRH_Roles": 25, "MinRolesRH_Edges": 754, "NewRolesRH_Roles": 53, "NewRolesRH_Edges": 786},
    "2level 05": {"MinRolesRH_Roles": 25, "MinRolesRH_Edges": 719, "NewRolesRH_Roles": 48, "NewRolesRH_Edges": 739},
    "2level 06": {"MinRolesRH_Roles": 25, "MinRolesRH_Edges": 729, "NewRolesRH_Roles": 59, "NewRolesRH_Edges": 772},
    "2level 07": {"MinRolesRH_Roles": 25, "MinRolesRH_Edges": 992, "NewRolesRH_Roles": 43, "NewRolesRH_Edges": 998},
    "2level 08": {"MinRolesRH_Roles": 25, "MinRolesRH_Edges": 890, "NewRolesRH_Roles": 55, "NewRolesRH_Edges": 925},
    "2level 09": {"MinRolesRH_Roles": 50, "MinRolesRH_Edges": 1097, "NewRolesRH_Roles": 94, "NewRolesRH_Edges": 1144},
    "2level 10": {"MinRolesRH_Roles": 50, "MinRolesRH_Edges": 1490, "NewRolesRH_Roles": 110, "NewRolesRH_Edges": 1564},
}

benchmarks, role_ratios, edge_ratios = [], [], []
for name, vals in data.items():
    mr_r, mr_e = vals["MinRolesRH_Roles"], vals["MinRolesRH_Edges"]
    nr_r, nr_e = vals["NewRolesRH_Roles"], vals["NewRolesRH_Edges"]
    if mr_r == 0 or mr_e == 0:
        continue
    benchmarks.append(name)
    role_ratios.append(nr_r / mr_r)
    edge_ratios.append(nr_e / mr_e)

benchmarks, role_ratios, edge_ratios = zip(*sorted(zip(benchmarks, role_ratios, edge_ratios), key=lambda x: x[0]))

x = np.arange(len(benchmarks))

plt.figure(figsize=(14, 6))
plt.plot(x, role_ratios, linestyle='-', color='k', label="Role Ratio", linewidth=2)
plt.plot(x, edge_ratios, linestyle='--', color='k', label="Edge Ratio", linewidth=2)

plt.axhline(1, color="gray", linestyle="-", linewidth=0.5)
plt.xticks(x, benchmarks, rotation=90, fontsize=7)
plt.ylabel("Ratio")
plt.xlabel("Benchmark", fontsize=10)
# plt.title("Line Graph of Role and Edge Ratios: NewRolesRH vs MinRolesRH")
plt.legend()
plt.tight_layout()
plt.show()

print(f"{'Benchmark':<15} {'RoleRatio':>10} {'EdgeRatio':>10}")
for b, r, e in zip(benchmarks, role_ratios, edge_ratios):
    print(f"{b:<15} {r:10.2f} {e:10.2f}")
