# Plots Data
import matplotlib.pyplot as plt
from IPython import display


plt.ion()
def plot(scores, mean_scores, avg10=None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    # avg10
    if avg10 != None:
        plt.plot(avg10)
        plt.text(len(avg10)-1, avg10[-1], str(avg10[-1]))
    plt.show(block=False)
    plt.pause(.1)