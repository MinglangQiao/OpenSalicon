# encoding=utf-8
from video_subject import f_game_dic_new_all, subjects_dic_all, f_game_dic_new_test, f_dic_train, f_dic_for_sauc


fov_width = 1080 # pixel
fov_heigth = 1200
fov_step = 20
saliency_width = int(fov_width / fov_step)
saliency_height = int(fov_heigth / fov_step)

video_dic = f_game_dic_new_all
train_dic = []
subject_dic = subjects_dic_all
video_test = f_game_dic_new_test


source_path = 'I:\\2_0925_全景视频的眼动研究\\VR_杨燕丹\\filtered_Data'
