import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc

sns.set_style('darkgrid')
x = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


# DSM + Grad
dsm_grad_i_pos = [85.02, 60.31, 50.35, 45.25, 44.2, 43.39, 42.43, 41.04, 32.73]
dsm_grad_i_neg = [85.02, 85.11, 84.83, 84.02, 83.43, 83.15, 82.01, 79.39, 32.73]
dsm_grad_t_pos = [85.02, 46.12, 31.66, 18.48, 12.84, 10.31, 6.3, 5.84, 5.83]
dsm_grad_t_neg = [85.02, 80.47, 70.05, 40.85, 28.14, 21.02, 7.24, 5.83, 5.83]

# DSM + Grad + Cam
dsm_grad_cam_i_pos = [85.02, 60.56, 52.07, 46.94, 45.75, 44.83, 43.88, 41.72, 32.73]
dsm_grad_cam_i_neg = [85.02, 85.13, 84.83, 83.3, 82.52, 81.55, 79.88, 76.38, 32.73]
dsm_grad_cam_t_pos = [85.02, 56.5, 40.0, 22.5, 16.68, 12.99, 6.46, 5.83, 5.83]
dsm_grad_cam_t_neg = [85.02, 74.11, 60.04, 35.4, 24.35, 18.13, 6.88, 5.83, 5.83]

# DSM
dsm_i_pos = [85.02, 63.13, 58.34, 53.56, 52.56, 51.67, 50.76, 49.37, 32.73]
dsm_i_neg = [85.02, 84.37, 82.75, 79.53, 78.83, 76.89, 74.11, 56.48, 32.73]
dsm_t_pos = [85.02, 63.48, 46.78, 29.58, 20.05, 14.81, 6.58, 5.87, 5.83]
dsm_t_neg = [85.02, 66.45, 53.12, 32.02, 21.73, 16.77, 7.24, 5.83, 5.83]

# RM
hcrm_i_pos = [85.02, 59.16, 51.58, 47.34, 46.88, 46.27, 44.61, 42.66, 32.73] 
R_t_i = [85.02, 61.75, 54.1, 48.89, 47.38, 46.64, 44.84, 42.22, 32.73]
hcrm_i_neg = [85.02, 85.07, 84.91, 84.04, 83.79, 83.18, 82.16, 79.38, 32.73] 
R_t_i = [85.02, 85.04, 84.69, 83.66, 83.2, 82.28, 80.42, 76.76, 32.73]
hcrm_t_pos = [85.02, 43.6, 26.48, 14.37, 11.39, 9.28, 6.28, 5.83]
hcrm_t_neg = [85.02, 81.34, 72.81, 49.69, 35.57, 26.33, 7.97, 5.87]

# Transformer Attr


# Raw Attn
raw_attn_i_pos = [85.02, 61.48, 51.74, 46.28, 45.52, 44.14, 42.65, 40.39, 32.73]
raw_attn_i_neg = [85.02, 85.16, 84.64, 83.23, 82.66, 82.03, 80.43, 76.98, 32.73]
raw_attn_t_pos = [85.02, 54.17, 32.78, 16.34, 12.87, 10.09, 6.37, 5.83]
raw_attn_t_neg = [85.02, 75.92, 61.16, 40.73, 28.93, 22.66, 7.77, 5.86]

# Gradcam
gradcam_i_pos = [85.02, 80.23, 74.82, 63.66, 60.94, 57.93, 54.76, 50.23, 32.73]
gradcam_i_neg = [85.02, 82.38, 75.73, 67.31, 65.72, 63.63, 61.87, 57.76, 32.73]
gradcam_t_pos = [85.02, 67.03, 52.67, 37.08, 25.93, 20.08, 7.19, 5.84, 5.83]
gradcam_t_neg = [85.02, 69.46, 54.07, 35.26, 24.77, 19.71, 7.3, 5.83, 5.83]

# Rollout
rollout_i_pos = [85.02, 67.12, 57.14, 51.4, 50.43, 49.55, 48.27, 46.52, 32.73]
rollout_i_neg = [85.02, 85.08, 84.67, 83.62, 83.32, 82.64, 80.57, 68.34, 32.73]
rollout_t_pos = [85.02, 65.15, 47.88, 22.33, 14.96, 11.55, 6.37, 5.83, 5.83]
rollout_t_neg = [85.02, 70.15, 51.65, 38.21, 27.75, 19.4, 6.5, 5.83, 5.83]



plt.title('Positive perturbation test on image modality')
plt.plot(x, dsm_i_pos)
plt.plot(x, dsm_grad_i_pos)
plt.plot(x, dsm_grad_cam_i_pos)
plt.plot(x, hcrm_i_pos)
# plt.plot(x, transformer_attr_i_pos)
plt.plot(x, raw_attn_i_pos)
# plt.plot(x, partial_lrp_i_pos)
plt.plot(x, gradcam_i_pos)
plt.plot(x, rollout_i_pos)




plt.legend(['DSM (AUC: ' + str(round(auc(x, dsm_i_pos), 2)) + ')', 
            'DSM + grad (' + str(round(auc(x, dsm_grad_i_pos), 2)) + ')',
            'DSM + grad + attn. (' + str(round(auc(x, dsm_grad_cam_i_pos), 2)) + ')', 
            'Relevance maps (' + str(round(auc(x, hcrm_i_pos), 2)) + ')', 
            # 'Transformer attribution (' + str(round(auc(x, transformer_attr_i_pos), 2)) + ')', 
            'Raw attention (' + str(round(auc(x, raw_attn_i_pos), 2)) + ')', 
            # 'LRP (' + str(round(auc(x, partial_lrp_i_pos), 2)) + ')',
            'GradCAM (' + str(round(auc(x, gradcam_i_pos), 2)) + ')', 
            'Rollout (' + str(round(auc(x, rollout_i_pos), 2)) + ')'
          ]) 


# plt.title('Negative perturbation test on image modality')
# plt.plot(x, dsm_i_neg)
# plt.plot(x, dsm_grad_i_neg)
# plt.plot(x, dsm_grad_cam_i_neg)
# plt.plot(x, hcrm_i_neg)
# # plt.plot(x, transformer_attr_i_neg)
# plt.plot(x, raw_attn_i_neg)
# # plt.plot(x, partial_lrp_i_neg)
# plt.plot(x, gradcam_i_neg)
# plt.plot(x, rollout_i_neg)



# plt.legend([ 'DSM (AUC: ' + str(round(auc(x, dsm_i_neg), 2)) + ')', 
#             'DSM + grad (' + str(round(auc(x, dsm_grad_i_neg), 2)) + ')', 
#             'DSM + grad + attn. (' + str(round(auc(x, dsm_grad_cam_i_neg), 2)) + ')', 
#             'Relevance maps (' + str(round(auc(x, hcrm_i_neg), 2)) + ')', 
#             # 'Transformer attribution (' + str(round(auc(x, transformer_attr_i_neg), 2)) + ')', 
#             'Raw attention (' + str(round(auc(x, raw_attn_i_neg), 2)) + ')', 
#             # 'LRP (' + str(round(auc(x, partial_lrp_i_neg), 2)) + ')',
#             'GradCAM (' + str(round(auc(x, gradcam_i_neg), 2)) + ')', 
#             'Rollout (' + str(round(auc(x, rollout_i_neg), 2)) + ')'
#           ]) 



# plt.title('Positive perturbation test on text modality')
# plt.plot(x, dsm_t_pos)
# plt.plot(x, dsm_grad_t_pos)
# plt.plot(x, dsm_grad_cam_t_pos)
# plt.plot(x, hcrm_t_pos)
# plt.plot(x, transformer_attr_t_pos)
# plt.plot(x, raw_attn_t_pos)
# plt.plot(x, partial_lrp_t_pos)
# plt.plot(x, gradcam_t_pos)
# plt.plot(x, rollout_t_pos)



# plt.legend([ 'DSM (AUC: '+ str(round(auc(x, dsm_t_pos), 2)) + ')', 
#             'DSM + grad (' + str(round(auc(x, dsm_grad_t_pos), 2)) + ')', 
#             'DSM + grad + attn. (' + str(round(auc(x, dsm_grad_cam_t_pos), 2)) + ')', 
#             'Relevance maps (' + str(round(auc(x, hcrm_t_pos), 2)) + ')', 
#             'Transformer attribution (' + str(round(auc(x, transformer_attr_t_pos), 2)) + ')', 
#             'Raw attention (' + str(round(auc(x, raw_attn_t_pos), 2)) + ')', 
#             'LRP (' + str(round(auc(x, partial_lrp_t_pos), 2)) + ')',
#             'GradCAM (' + str(round(auc(x, gradcam_t_pos), 2)) + ')', 
#             'Rollout (' + str(round(auc(x, rollout_t_pos), 2)) + ')'
#           ]) 


# plt.title('Negative perturbation test on text modality')
# plt.plot(x, dsm_t_neg)
# plt.plot(x, dsm_grad_t_neg)
# plt.plot(x, dsm_grad_cam_t_neg)
# plt.plot(x, hcrm_t_neg)
# plt.plot(x, transformer_attr_t_neg)
# plt.plot(x, raw_attn_t_neg)
# plt.plot(x, partial_lrp_t_neg)
# plt.plot(x, gradcam_t_neg)
# plt.plot(x, rollout_t_neg)



# plt.legend(['DSM (AUC: ' + str(round(auc(x, dsm_t_neg), 2)) + ')', 
#             'DSM + grad (' + str(round(auc(x, dsm_grad_t_neg), 2)) + ')', 
#             'DSM + grad + attn. (' + str(round(auc(x, dsm_grad_cam_t_neg), 2)) + ')', 
#             'Relevance maps (' + str(round(auc(x, hcrm_t_neg), 2)) + ')', 
#             'Transformer attribution (' + str(round(auc(x, transformer_attr_t_neg), 2)) + ')', 
#             'Raw attention (' + str(round(auc(x, raw_attn_t_neg), 2)) + ')', 
#             'LRP (' + str(round(auc(x, partial_lrp_t_neg), 2)) + ')',
#             'GradCAM (' + str(round(auc(x, gradcam_t_neg), 2)) + ')', 
#             'Rollout (' + str(round(auc(x, rollout_t_neg), 2)) + ')'
#           ]) 


plt.xlabel('Fraction of tokens removed')
plt.ylabel('Accuracy')
plt.show()