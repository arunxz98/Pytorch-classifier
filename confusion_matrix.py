#Helps create a simpler visual represenataion of the models accuracy on different
#classes

import pandas as pd
from pycm import *
from matplotlib import pyplot as plt


df = pd.read_csv("pred_outputs_without_weights.csv")

gt_class = df["class_name"].to_list()
pred_class = df["pred_class_name"].to_list()

cm = ConfusionMatrix(actual_vector=gt_class,predict_vector=pred_class,classes=["Class 0","Class 1","Class 2","Class 3"])
cm.print_matrix()
cm.plot(cmap=plt.cm.Greens,number_label=True,plot_lib="matplotlib")
plt.savefig("cfm_without_weights.png")
