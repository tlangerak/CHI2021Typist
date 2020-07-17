import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv("grades.csv", header=None)
    final_score = pd.DataFrame(0, index=data.index, columns=data.index)
    final_flipped = pd.DataFrame(None, index=list(range(data.shape[0] ** 2)), columns=["score", "s1", "s2"])
    n_questions = data.shape[1]
    idx = 0
    for index1, row1 in data.iterrows():
        for index2, row2 in data.iterrows():
            simscore = 0
            if index1 < index2:
                for col in range(1, n_questions):
                    if (float(row1[col]) - float(row2[col]))**2 <= 0.5**2 and row1[col] != 0 and row2[col] != 0:
                        simscore += 1
                if simscore <10:
                    simscore = None

                final_score[index1][index2] = simscore
                final_flipped.at[idx, "score"] = simscore
                final_flipped.at[idx, "s1"] = row1[0]
                final_flipped.at[idx, "s2"] = row2[0]

            else:
                final_score[index1][index2] = None
                final_flipped.at[idx, "score"] = simscore
                final_flipped.at[idx, "s1"] = row1[0]
                final_flipped.at[idx, "s2"] = row2[0]
            idx += 1

    final_flipped = final_flipped.sort_values(by=['score'], ascending=False)
    final_flipped.to_csv("foo4.csv", index=False)
    fig, ax = plt.subplots()

    ax.imshow(final_score)
    print(data.iloc[:, 0])
    print(final_score)
    ax.set_xticks(np.arange(final_score.shape[0]))
    ax.set_yticks(np.arange(final_score.shape[1]))
    ax.set_xticklabels(data.iloc[:, 0], fontdict={'fontsize': 2})
    ax.set_yticklabels(data.iloc[:, 0], fontdict={'fontsize': 2})
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    data = final_score.values
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if y > x and data[y, x] > 0:
                plt.text(x, y, '{}'.format(int(data[y, x])),
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontdict={'fontsize': 1.5},
                         color='w')
    fig.savefig("foo4.pdf", papertype='a4')
    plt.show()
