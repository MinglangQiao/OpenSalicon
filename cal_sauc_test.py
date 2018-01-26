#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy.io as sio
from suppor_lib import get_subjects
from config import video_dic, video_test, f_dic_train, f_dic_for_sauc
from support import get_video_config, save_txt, read_txt
from suppor_lib import haversine
import math
import os
from config import model_name_without_fcb, model_name_with_fcb
from PIL import Image


class SAUC():
    """
    the class for cal sauc
    """

    def __init__(self):
        from config import f_dic_for_sauc

        self.N_subjects = 52
        self.NVideo = 60
        self.Frame_each_video = 10
        self.Nsplits = 100

    def get_total_config(self, video_name):
        from support import get_video_config

        '''load in mat data of head movement'''
        # matfn = '../../'+self.data_base+'/FULLdata_per_video_frame.mat'
        matfn = '/home/ml/video_data_mat.mat'
        data_all = sio.loadmat(matfn)
        # print('>>>>>>>>>>>>>: ', np.shape(data_all))
        self.env_id = video_name
        data = data_all[self.env_id]

        self.subjects_total, self.data_total, self.subjects, _ = get_subjects(data,0)
        print('>>>>>>>>>>>>>>debug1: ',self.subjects_total, self.data_total, self.subjects)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>subjects_total: "+str(self.subjects_total))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>data_total: "+str(self.data_total))

        '''init video and get paramters'''
        # video = cv2.VideoCapture('../../'+self.data_base+'/' + self.env_id + '.mp4')
        FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_name)

        # video = cv2.VideoCapture('/home/minglang/vr_new/'+self.env_id + '.mp4')
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>video: "+video)
        # video = cv2.VideoCapture('/home/minglang/vr_new/A380.mp4')
        self.frame_per_second = FRAMERATE
        self.frame_total = FRAMESCOUNT
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>frame_total: "+str(self.frame_total))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>self.env_id: "+str(self.env_id))
        self.video_size_width = int(IMAGEWIDTH)
        self.video_size_heigth = int(IMAGEHEIGHT)
        self.second_total = self.frame_total / self.frame_per_second
        self.data_per_frame = self.data_total / self.frame_total

        '''compute step lenth from data_tensity'''
        data_tensity = 10
        self.second_per_step = max(data_tensity/self.frame_per_second, data_tensity/self.data_per_frame/self.frame_per_second)
        self.frame_per_step = self.frame_per_second * self.second_per_step
        self.data_per_step = self.data_per_frame * self.frame_per_step

        '''compute step_total'''
        self.step_total = int(self.data_total / self.data_per_step) + 1

        print(">>>>>>>>>>>>>>>>>step_total: ", str(self.step_total))

    def prepare_data(self):
        pass

    def cal_sauc_one_video(self):
        pass

    def calc_score(self, gtsAnn, resAnn, shufMap, stepSize=.01):
        """
        Computer SAUC score. A simple implementation
        :param gtsAnn : list of fixation annotataions, lile: [[1,2], [2,3], ... ]
        :param resAnn : list only contains one element: the result annotation - predicted saliency map, (binary matrix)
        :return score: int : score
        """

        salMap = resAnn - np.min(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.max(salMap) # normalization
        Sth = np.asarray([ salMap[y-1][x-1] for y,x in gtsAnn ]) # the location difference in (1,1) to (0,0), get the fixation location's val in heatmap
        Nfixations = len(gtsAnn)

        others = np.copy(shufMap)
        for y,x in gtsAnn:
            others[y-1][x-1] = 0 # delet the congtu points

        ind = np.nonzero(others) # find fixation locations on other images
        nFix = shufMap[ind] # each valye should be 1
        randfix = salMap[ind] # get the get the  negative fixation location's val in heatmap
        Nothers = sum(nFix) # number of fixations

        allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
        allthreshes = allthreshes[::-1]
        tp = np.zeros(len(allthreshes)+2)
        fp = np.zeros(len(allthreshes)+2)
        tp[-1]=1.0
        fp[-1]=1.0
        tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
        fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

        auc = np.trapz(tp,fp)
        return auc

    def get_raw_data(self, video_name):
        from config import subject_dic

        one_video_lon = []
        one_video_lat = []

        for i in range(len(subject_dic)):

            file = './filtered_Data' + '//' + str(subject_dic[i]) + '//' + str(video_name) + '.txt'
            f = open(file)
            lines = f.readlines()
            raw_data = []  # notice to reset it

            for line in lines:
                line = line.split()
                raw_data.append(line)
            raw_data = np.array(raw_data)

            head_lon = raw_data[:, 2]
            head_lat = raw_data[:, 1]

            one_video_lon.append(head_lon)
            one_video_lat.append(head_lat)

        return one_video_lon, one_video_lat

    def get_random_fixation_location(self):
        """
        get the shuffed map's fixation location
        """
        import random
        from config import video_dic, video_test, f_dic_train, f_dic_for_sauc
        from support import get_video_config, save_txt, read_txt

        NVideo = 60
        Frame_each_video = 10
        Nsplits = 100
        N_subjects = 52

        '1_fix the seed, uncommnet this until 8'
        # np.random.seed(0)
        # random.seed(0)
        '2_get video index'
        ind_video = np.arange(0, NVideo)
        np.random.shuffle(ind_video)

        # print('>>>>>>>>>>>> ind_frame: ', ind_frame)

        '3_get the train_dic'
        # print([x for x in video_dic if x not in video_test])

        'get sauc_video'
        # for i in ind_video:
        #     print("'%s',"%f_dic_train[i])

        'get all locations'
        # Total_frame_lon = []
        # Total_frame_lat = []
        # for i_video in range(NVideo):
        #     one_video_lon, one_video_lat = self.get_raw_data(f_dic_for_sauc[i_video])
        #     one_video_lon = np.array(one_video_lon)
        #     one_video_lat = np.array(one_video_lat)
        #
        #     FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(f_dic_for_sauc[i_video])
        #     'get frame_index'
        #     ind_frame = random.sample(range(0, FRAMESCOUNT * 2), Frame_each_video)
        #
        #     for i_fame in ind_frame:
        #         print('>>>>>>>>>>>>sample: %s, %s/%d'%(np.shape(one_video_lon[:, 1]), i_fame, FRAMESCOUNT))
        #         Total_frame_lon.extend(one_video_lon[:, i_fame])
        #         Total_frame_lat.extend(one_video_lat[:, i_fame])
        #
        #         save_path_lon = 'random_fix/' + 'total_fixation_lon.txt'
        #         save_path_lat = 'random_fix/' + 'total_fixation_lat.txt'
        #
        #         save_txt(Total_frame_lon, save_path_lon)
        #         save_txt(Total_frame_lat, save_path_lat)
        #
        #     print('>>>>>>>>>>>vide[%d]'%i_video)
        #
        # print('>>>>>>>>>>>np.shape(Total_frame_lat): ',np.shape(Total_frame_lat))
        #
        # save_path_lon = 'random_fix/' + 'total_fixation_lon.txt'
        # save_path_lat = 'random_fix/' + 'total_fixation_lat.txt'
        #
        # save_txt(Total_frame_lon, save_path_lon)
        # save_txt(Total_frame_lat, save_path_lat)

        'read data and plot'
        save_path_lon = 'random_fix/' + 'total_fixation_lon.txt'
        save_path_lat = 'random_fix/' + 'total_fixation_lat.txt'

        Total_frame_lon = read_txt(save_path_lon)
        print('>>>>>>>>>>>>>>> Total_frame_lon : ', len(Total_frame_lon[0]))
        Total_frame_lat = read_txt(save_path_lat)

        # print(np.shape(Total_frame_lat))
        # plt.scatter(Total_frame_lon, Total_frame_lat)
        # plt.show()
        # plt.savefig("examples.jpg")

        '8_get the splits index, uncomment the np.random.seed(0) in former'
        for i_splits in range(Nsplits):
            # np.random.seed(i_splits)
            ind_splits = random.sample(range(0, NVideo * Frame_each_video * N_subjects), N_subjects)
            print('>>>>>>>>>>>>11: ', len(Total_frame_lon), ind_splits)

            save_path_lon = 'random_fix/' + 'lon_splits' + str(i_splits) + '.txt'
            save_path_lat = 'random_fix/' + 'lat_splits' + str(i_splits) + '.txt'

            data_splits_lon = [ Total_frame_lon[0][i] for i in ind_splits]
            data_splits_lat = [ Total_frame_lat[0][i] for i in ind_splits]
            save_txt(data_splits_lon, save_path_lon)
            save_txt(data_splits_lat, save_path_lat)

            plt.scatter(data_splits_lon, data_splits_lat)
            plt.xlim(-180, 180)
            plt.ylim(-90, 90)
            plt.axis('on')
            # plt.show()
            plt.savefig("random_fix/splits_" + str(i_splits) + "_fixations.png")
            plt.close()

    def save_gt_heatmaps(self):
        print('>>>>>>>>>: save_gt_heatmaps')

        '''for fixation'''
        # sigma = 51.0 / (math.sqrt(-2.0*math.log(0.5)))
        sigma = 7 #cc is large .chose half of sigma
        groundtruth_heatmaps = []

        for step in range(self.step_total):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            try:
                groundtruth_fixation = np.zeros((self.subjects_total, 2))
                for subject in range(self.subjects_total):
                    # print("self.subjects_total: ",self.subjects_total)
                    # print(s_qiao)
                    groundtruth_fixation[subject, 0] = self.subjects[subject].data_frame[data].p[0]
                    groundtruth_fixation[subject, 1] = self.subjects[subject].data_frame[data].p[1]
                groundtruth_heatmap = self.fixation2salmap_sp_my_sigma(groundtruth_fixation, self.salmap_width, self.salmap_height, my_sigma = sigma)
                self.save_heatmap(heatmap=groundtruth_heatmap,
                                  path='/home/minglang/PAMI/test_file/ground_truth_hmap/',
                                  name=str(step))
                groundtruth_heatmaps += [groundtruth_heatmap]
                print(np.shape(groundtruth_heatmaps))
            except Exception  as e:
                print(Exception,":",e)
                continue
        print(s)

    def get_ground_fixation(self, video_name, frame):

        one_video_lon, one_video_lat = self.get_raw_data(video_name)
        one_video_lon = np.array(one_video_lon)
        one_video_lat = np.array(one_video_lat)

        lon_one_frame = one_video_lon[:, frame * 20] # step is 10 but sample rate is 2 * framerate
        lat_one_frame = one_video_lat[:, frame * 20]
        'left up is positive'
        lon_one_frame = [np.round(-float(i)+ 180).astype('int') for i in lon_one_frame]
        lat_one_frame = [np.round(-float(i)+ 90).astype('int') for i in lat_one_frame]
        gt_location = []

        for i in range(len(lon_one_frame)):
            gt_location.append([lat_one_frame[i], lon_one_frame[i]])

        'for debug'
        ground_hmap = np.zeros((180, 360))
        for i in range(self.N_subjects):
            if lon_one_frame[i] >= 360:
                lon_one_frame[i] = 359
            if lat_one_frame[i] >= 180:
                lat_one_frame[i] = 179
            ground_hmap[ lat_one_frame[i] ][ lon_one_frame[i] ] = 1

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_title('ground_fixation_hmap')
        # plt.imshow(ground_hmap)
        # plt.show()

        # print('>>>>>>>>>>>>>>>>>: ', gt_location)

        return gt_location, ground_hmap

    def get_predict_hmap(self, path):
        "ld"
        img = imageio.imread(path)

        # print(np.shape(img))
        # print('>>>>>>>>>>>>>>>： ', (img))
        return img

    def get_shuffed_map(self, i_splits):
        from support import read_txt

        save_path_lon = 'random_fix/' + 'lon_splits' + str(i_splits) + '.txt'
        save_path_lat = 'random_fix/' + 'lat_splits' + str(i_splits) + '.txt'

        data_splits_lon = read_txt(save_path_lon)
        data_splits_lat = read_txt(save_path_lat)

        data_splits_lon = [(np.round(i)).astype('int') for i in data_splits_lon]
        data_splits_lat = [(np.round(i)).astype('int') for i in data_splits_lat]

        # 'convert to left to is (0, 0)'
        data_splits_lon = np.array([(-i + 180) for i in data_splits_lon])
        data_splits_lat = np.array([(-i  + 90) for i in data_splits_lat])
        # print('>>>>>> data_splits_lon: ', data_splits_lon,  data_splits_lat)

        salMap = np.zeros((180, 360))
        for i in range(self.N_subjects):
            if data_splits_lon[0][i] >= 360:
                data_splits_lon[0][i] = 359
            if data_splits_lat[0][i] >= 180:
                data_splits_lat[0][i] = 179
            salMap[ data_splits_lat[0][i] ][ data_splits_lon[0][i] ] = 1

        # plt.imshow(salMap)
        # plt.xlim(180, 180)
        # plt.ylim(-90, 90)
        # plt.show()
        # print('>>>>>>>>>>> data_splits_lon', salMap)

        # plt.scatter(data_splits_lon, data_splits_lat)
        # plt.xlim(-180, 180)
        # plt.ylim(-90, 90)
        # plt.axis('on')
        # plt.show()

        # print('>>>>>>>> 1', ind_frame)
        return salMap

    def fixation2salmap_sp_my_sigma(self,fixation, mapwidth, mapheight, my_sigma = 7.0):
        """
        get the hmap, note the up left is positive
        """
        fixation_total = np.shape(fixation)[0]
        x_degree_per_pixel = 360.0 / mapwidth
        y_degree_per_pixel = 180.0 / mapheight
        salmap = np.zeros((mapwidth, mapheight))
        for x in range(mapwidth):
            for y in range(mapheight):
                cur_lon = 1 * ( -x * x_degree_per_pixel + 180.0 )
                cur_lat = 1 * ( -y * y_degree_per_pixel + 90.0 )
                for fixation_count in range(fixation_total):
                    cur_fixation_lon = fixation[fixation_count][0]
                    cur_fixation_lat = fixation[fixation_count][1]
                    distance_to_cur_fixation = haversine(lon1=cur_lon,
                                                         lat1=cur_lat,
                                                         lon2=cur_fixation_lon,
                                                         lat2=cur_fixation_lat)
                    distance_to_cur_fixation = distance_to_cur_fixation / math.pi * 180.0
                    sal = math.exp(-1.0 / 2.0 * (distance_to_cur_fixation**2) / (my_sigma**2))
                    salmap[x, y] += sal
        salmap = salmap * ( 1.0 / np.amax(salmap) )
        salmap = np.transpose(salmap)
        return salmap

    def get_hmap_one_video(self, video_name, model_name):
        """
        get hmap of one video with step = 10, note that should frame step should
        be 20 as the sample rate is 2 * frame rate
        """
        FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_name)
        step_size = 10
        frame_step = 20
        step_total = int(FRAMESCOUNT/10)

        one_video_lon, one_video_lat = self.get_raw_data(video_name)
        one_video_lon = np.array(one_video_lon)
        one_video_lat = np.array(one_video_lat)

        print('>>>>>>>>>>>>>>>>>>>>>>>step_total: ', step_total)
        for i_step in range(0, step_total):

            if i_step >= 0:
                i_frame = i_step * frame_step

                one_frame_lon = [np.round(float(i)) for i in one_video_lon[:, i_frame]]
                one_frame_lat = [np.round(float(i)) for i in one_video_lat[:, i_frame]]

                fixation_data = []
                for i in range(len(one_frame_lon)):
                    if one_frame_lon[i] >= 360:
                        one_frame_lon[i] = 359
                    if one_frame_lat[i] >= 180:
                        one_frame_lon[i] = 179
                    fixation_data.append([one_frame_lon[i], one_frame_lat[i]])

                hmap = self.fixation2salmap_sp_my_sigma(fixation_data, 360, 180)
                # plt.subplot(211)
                # plt.scatter(one_frame_lon, one_frame_lat)
                # plt.xlim( 180, -180 )
                # plt.ylim( -90, 90 )
                # plt.subplot(212)
                # plt.imshow(hmap)
                # plt.show()

                self.save_heatmap(heatmap=hmap,
                                  path='/Data/Hmaps_0120/' + model_name,
                                  name= video_name + '_' + str(i_step))

                print('>>>>>>>>>>>>>>>>>>>>>>> processing finished: step%d/%d'%(i_step, step_total))


    def remap_for_reference_direction_bug(self):
        pass


    def cal_sauc_one_video(self, video_name, model_name):
        """
        cal sauc score of one video in on model
        """
        FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_name)
        step_size = 10
        frame_step = 20
        step_total = int(FRAMESCOUNT/step_size)

        one_video_lon, one_video_lat = self.get_raw_data(video_name)
        one_video_lon = np.array(one_video_lon)
        one_video_lat = np.array(one_video_lat)

        ave_sauc_one_video = []
        for i_step in range(step_total):
            if i_step >= 0: # for debug
                'get predicted map'
                hmap_path = 'Data/Hmaps_0120/' + model_name + '/' + video_name + '_' + str(i_step) +'.jpg'
                if model_name == model_name_without_fcb[5]:
                    hmap_path = 'Data/Hmaps_0120/' + model_name + '/_' + video_name + '_' + str(i_step) +'.png'
                predict_map = self.get_predict_hmap(hmap_path)

                if model_name == model_name_without_fcb[3] or model_name == model_name_with_fcb[3]:
                    predict_map = self.FZ(predict_map)

                'get ground fixation list'
                ground_fixation, ground_binary_map = self.get_ground_fixation(video_name, i_step)

                sauc = []
                for i_splits in range(self.Nsplits):
                    shuffed_map = self.get_shuffed_map(i_splits)
                    sauc_0 = self.calc_score(ground_fixation,  predict_map, shuffed_map)
                    sauc.append(sauc_0)
                    'for debug'
                    # plt.subplot(312)
                    # plt.imshow(predict_map)
                    # plt.subplot(313)
                    # plt.imshow(ground_binary_map)
                    # 'test rmap coordinate bug'
                    # hmap_path_4_path = 'Data/Hmaps_0120/' + model_name_without_fcb[4] + '/' + video_name + '_' + str(i_step) + '.jpg'
                    # hmap_path_4 = self.get_predict_hmap(hmap_path_4_path)
                    # hmap_path_4_1 = self.FZ(hmap_path_4)
                    # print(hmap_path_4)
                    # plt.subplot(311)
                    # plt.imshow(hmap_path_4_1)
                    # plt.show()
                    print('>>>>>sauc_splits:%d, score: %f: '%(i_splits, sauc_0))
                    # print(t)

                ave_sauc = np.mean(sauc)
                ave_sauc_one_video.append(ave_sauc)
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.>ave_sauc: ', ave_sauc)

        ave_sauc_one_video = np.mean(ave_sauc_one_video)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>final: ave_sauc_one_video: ', ave_sauc_one_video)

        return ave_sauc_one_video


    def get_fine_image(self, video_name):

        FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_name)
        step_size = 10
        frame_step = 20
        step_total = int(FRAMESCOUNT/step_size)

        one_video_lon, one_video_lat = self.get_raw_data(video_name)
        one_video_lon = np.array(one_video_lon)
        one_video_lat = np.array(one_video_lat)
        read_image_path0 = '/media/ml/Data1/2_0925_全景视频的眼动研究/Salience_of_FOV/程序/Finding_Content/' + video_name + '_raw_frames/'
        save_image_path0 = '/home/ml/OpenSALICON/Data/ml/fine_images/' + video_name

        ave_sauc_one_video = []
        for i_step in range(step_total):
            if i_step > 0: # for debug

                read_image_path =  read_image_path0 + '%03d'%( 10 * i_step) + '.png'
                save_image_path =  save_image_path0 + '_' + '%03d'%(i_step) + '.jpg'
                im = imageio.imread(read_image_path)
                imageio.imwrite(save_image_path, im)
                # plt.imshow(im)
                # plt.show()


    def get_fixation_map(self, video_name):

        FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_name)
        step_size = 10
        frame_step = 20
        step_total = int(FRAMESCOUNT/step_size)

        one_video_lon, one_video_lat = self.get_raw_data(video_name)
        one_video_lon = np.array(one_video_lon)
        one_video_lat = np.array(one_video_lat)
        read_image_path0 = '/home/ml/Pami_reponse_SAUC/Data/Hmaps_0120/groundtruth_0120/_' + video_name +'_'
        save_image_path0 = '/home/ml/OpenSALICON/Data/ml/fixation_images/' + video_name

        ave_sauc_one_video = []
        for i_step in range(step_total):
            if i_step > 0: # for debug

                read_image_path =  read_image_path0 + str(i_step) + '.png'
                save_image_path =  save_image_path0 + '_' + '%03d'%(i_step) + '.jpg'
                im = imageio.imread(read_image_path)
                imageio.imwrite(save_image_path, im)


    def fz(self, a):
        return a[::-1]
    def FZ(self, mat):
        return np.array(self.fz(list(map(self.fz, mat))))


    def save_heatmap(self,heatmap, path, name):
        path = '/home/ml/Pami_reponse_SAUC/' + path
        if os.path.exists(path) is False:
            os.mkdir(path)
        heatmap = heatmap * 255.0
        imageio.imwrite(path+ '/_'+name+'.png',heatmap)

    def save_score(self, score,  model_name, video_name):

        path = '/home/ml/Pami_reponse_SAUC/result/'
        if os.path.exists(path) is False:
            os.makedirs(path)

        save_path = path + '/sauc_' + model_name + '.txt'
        save_data = video_name + '\t' + str(score) + '\n'

        f = open(save_path, 'a')
        f.write(save_data)
        f.close()

    def run(self):
        print('>>>>>>>>>>>>test_sauc')
        from config import video_dic, video_test, f_dic_train, f_dic_for_sauc
        from support import get_video_config, save_txt, read_txt

        frame = 100
        i_splits = 1

        # self.get_shuffed_map(1)

        for i_video in range(len(video_test)):

            if i_video == 14:
                FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_test[i_video])
                print('>>>>>>>>>>>>>>>>> video, frmaes', video_test[i_video], FRAMESCOUNT)

                frame_per_step = 10
                total_step = int(FRAMESCOUNT / frame_per_step )

                for i_step in range(total_step):

                    if i_step == 82:
                        hmap_path = '/home/ml/Data/Hmaps/ours/ff_best_heatmaps_ours_without_fcb/' + video_test[i_video] + '_' + str(i_step) +'.jpg'
                        print('>>>>>>>>>>>>>>: ', hmap_path)
                        predict_map = self.get_predict_hmap( hmap_path )
                        ground_fixation = self.get_ground_fixation( video_test[i_video], i_step)

                        plt.subplot(211)
                        plt.imshow(predict_map)
                        plt.subplot(212)
                        plt.imshow(ground_fixation)
                        plt.show()

                        'cal the sauc'
                        sauc = []
                        for i_splits in range(self.Nsplits):
                            shuffed_map = self.get_shuffed_map(i_splits)
                            sauc_0 = self.calc_score(ground_fixation,  predict_map, shuffed_map)
                            sauc.append(sauc_0)

                            print('>>>>>sauc_splits:%d, score: : '%(i_splits, sauc_0))

                        ave_sauc = np.mean(sauc)
                        print('>>>>>>>>>>>. ave_sauc: ', ave_sauc)
                            # print('>>>>>>>>>>> run: ',sauc_0,  np.shape(predict_map), ground_fixation, np.shape(shuffed_map))

    def run2(self):

        for i_video in range(len(video_test)):
            print('>>>>>>>>>>>>>>>>>>>.video: %d/%d_%s'%(i_video, len(video_test), video_test[i_video]))

            video_name = video_test[i_video]
            model_name = 'ours_without_fcb'
            self.get_hmap_one_video(video_name, model_name)

    def run3(self):
        "cal sauc of one video"
        for i_video in range(len(video_test)):
            print('>>>>>>>>>>>>>>>>>>>.video: %d/%d_%s'%(i_video, len(video_test), video_test[i_video]))

            if i_video >= 0:
                video_name = video_test[i_video]
                model_name = model_name_with_fcb[0]

                sauc = self.cal_sauc_one_video(video_name, model_name)
                self.save_score(sauc, model_name, video_name)

    def run4(self):
        for i_video in range(0, 10):
            print('>>>>>>>>>>>>>>>>>>>.video: %d/%d_%s'%(i_video, len(video_test), video_test[i_video]))

            if i_video >= 0:
                # self.get_fine_image(video_test[i_video])
                self.get_fixation_map(video_test[i_video])





if __name__ == '__main__':

    cal_sauc = SAUC()
    cal_sauc.run4()
