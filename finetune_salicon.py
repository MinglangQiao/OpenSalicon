import numpy as np
from PIL import Image
import pdb
import matplotlib.pyplot as plt
import sys
import time
from config import video_test
from support import get_video_config

class Fine_tune():
    'prepare train data and fine tune'

    def __init__(self):
        self.training_data_path = '/home/ml/OpenSALICON/Data/ml/' # PATH TO YOUR TRAINING DATA
        sys.path.insert(0, '/home/ml/caffe-master/python') # UPDATE YOUR CAFFE PATH HERE
        import caffe
        self.caffe = caffe
        caffe.set_mode_gpu()
        caffe.set_device(0)

        # load the solver
        self.solver = self.caffe.SGDSolver('solver_new.prototxt')
        self.solver.net.copy_from('salicon_osie.caffemodel') # untrained.caffemodel

    # def get_train_data_for_ml(self):
    #     """
    #     get the train data of minglang's project
    #     """
    #     fine_imgs = []
    #     coarse_imgs = []
    #     fix_imgs = []
    #     MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
    #     MEAN_VALUE = MEAN_VALUE[:,None, None]
    #
    #     for i in range(0, 1):
    #         video_name = video_test[i]
    #         FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_name)
    #         step_size = 10
    #         frame_step = 20
    #         step_total = int(FRAMESCOUNT/step_size)
    #
    #         for i in range(1, step_total-10):
    #             im = np.array(Image.open(self.training_data_path + 'coarse_images/' + video_name + '_' + '%03d'%i + '.jpg'), dtype=np.float32) # in RGB
    #             # put channel dimension first
    #             im = np.transpose(im, (2,0,1))
    #             # switch to BGR
    #             im = im[::-1, :, :]
    #             # subtract mean
    #             im = im - MEAN_VALUE
    #             im = im[None,:]
    #             assert(im.shape == (1,3,600,800))
    #             # TEST - CONVERT TO DOUBLE
    #             im = im / 255
    #             im = im.astype(np.dtype(np.float32))
    #             coarse_imgs.append(im)
    #             print '>>>>>>>>>>>>loading coarse_images: ' + str(i)
    #
    #         # now do the fine images
    #         for i in range(1, step_total-10):
    #             im = np.array(Image.open(self.training_data_path + 'fine_images/' + video_name + '_' + '%03d'%i + '.jpg'), dtype=np.float32) # in RGB
    #             # put channel dimension first
    #             im = np.transpose(im, (2,0,1))
    #             # switch to BGR
    #             im = im[::-1, :, :]
    #             # subtract mean
    #             im = im - MEAN_VALUE
    #             im = im[None,:]
    #             assert(im.shape == (1,3,1200,1600))
    #             # TEST - CONVERT TO DOUBLE
    #             im = im / 255
    #             im = im.astype(np.dtype(np.float32))
    #             fine_imgs.append(im)
    #             print '>>>>>>>>>>>>loading fine_images: ' + str(i)
    #
    #         # load fixations
    #         for i in range(1, step_total-10):
    #             im = np.array(Image.open(self.training_data_path + 'fixation_images/' + video_name + '_' +'%03d'%i + '.jpg'), dtype=np.float32)
    #             im = im[None,None,:]
    #             assert(im.shape == (1,1,38,50))
    #             # TEST - CONVERT TO DOUBLE
    #             im = im / 255
    #             im = im.astype(np.dtype(np.float32))
    #             fix_imgs.append(im)
    #             print '>>>>>>>>>>>>loading fixation_images: ' + str(i)
    #
    #         assert(len(fix_imgs) == len(fine_imgs) and len(fine_imgs) == len(coarse_imgs))
    #         print '>>>>>>>>>>>>>>>>>>>>>>>>>>> num_fix_imgs: ' + str(len(fix_imgs))
    #
    #         return coarse_imgs, fine_imgs, fix_imgs

    def get_one_batch_data(self, coarse_image_dic, fine_image_dic, fixation_image_dic, start_frame, batch_size = 50):
        """
        get one batch data for traing
        batch_size set to 50
        """
        fine_imgs = []
        coarse_imgs = []
        fix_imgs = []
        MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
        MEAN_VALUE = MEAN_VALUE[:,None, None]

        for i in range(start_frame, start_frame +  batch_size):
            'get coarse data'
            im = np.array(Image.open(coarse_image_dic[i]), dtype=np.float32) # in RGB
            # put channel dimension first
            im = np.transpose(im, (2,0,1))
            # switch to BGR
            im = im[::-1, :, :]
            # subtract mean
            im = im - MEAN_VALUE
            im = im[None,:]
            assert(im.shape == (1,3,600,800))
            # TEST - CONVERT TO DOUBLE
            im = im / 255
            im = im.astype(np.dtype(np.float32))
            coarse_imgs.append(im)
            print '>>>>>>>>>>>>loading coarse_images: ' + str(i)

        for i in range(start_frame, start_frame +  batch_size):
            'get fine data'
            im = np.array(Image.open(fine_image_dic[i]), dtype=np.float32) # in RGB
            # put channel dimension first
            im = np.transpose(im, (2,0,1))
            # switch to BGR
            im = im[::-1, :, :]
            # subtract mean
            im = im - MEAN_VALUE
            im = im[None,:]
            assert(im.shape == (1,3,1200,1600))
            # TEST - CONVERT TO DOUBLE
            im = im / 255
            im = im.astype(np.dtype(np.float32))
            fine_imgs.append(im)
            print '>>>>>>>>>>>>loading fine_images: ' + str(i)

        for i in range(start_frame, start_frame +  batch_size):
            'get fixation data'
            # load fixations
            im = np.array(Image.open(fixation_image_dic[i]), dtype=np.float32)
            im = im[None,None,:]
            assert(im.shape == (1,1,38,50))
            # TEST - CONVERT TO DOUBLE
            im = im / 255
            im = im.astype(np.dtype(np.float32))
            fix_imgs.append(im)
            print '>>>>>>>>>>>>loading fixation_images: ' + str(i)

        assert(len(fix_imgs) == len(fine_imgs) and len(fine_imgs) == len(coarse_imgs))
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>> num_fix_imgs: ' + str(len(fix_imgs))

        return coarse_imgs, fine_imgs, fix_imgs

    def train(self, coarse_imgs, fine_imgs, fix_imgs, i_batch, batch_size = 50):
        """
        fine tune the model
        """
        for i in range(0, len(coarse_imgs)):
            print 'working on ' + str(i) + ' of ' + str(len(coarse_imgs)) + ' batch: ' + str(i_batch) + ' of ' + str(int(15491/ batch_size))

            fine_img_to_process = fine_imgs[i]
            coarse_img_to_process = coarse_imgs[i]
            fix_img_to_process = fix_imgs[i]

            self.solver.net.blobs['fine_scale'].data[...] = fine_img_to_process
            self.solver.net.blobs['coarse_scale'].data[...] = coarse_img_to_process
            self.solver.net.blobs['ground_truth'].data[...] = fix_img_to_process
            self.solver.step(1)

        if i_batch % 10 == 0:
            self.solver.net.save('train_output/finetuned_salicon_{}.caffemodel'.format(i_batch))


    def get_image_list_of_jiang(self, path):
        """
        get random lis of jiang's data
        """
        import glob, random

        # random.seed(0)
        coarse_image_path = path + 'coarse_images/'
        fine_image_path = path + 'fine_images/'
        fixation_image_path = path + 'fixation_images/'

        all_coarse_images = glob.glob(coarse_image_path + '*.jpg')
        all_fine_images = glob.glob(fine_image_path + '*.jpg')
        all_fixation_images = glob.glob(fixation_image_path + '*.jpg')
        assert(len(all_coarse_images) == len(all_fine_images) and len(all_fine_images) == len(all_fixation_images))
        print '>>>>>>>>>>>>>>>total frame: ', str(len(all_coarse_images))

        'get random dic'
        random_images_index = random.sample(range(0, len(all_fine_images)), len(all_fine_images))
        random_coarse_image_dic = [all_coarse_images[i] for i in random_images_index]
        random_fine_image_dic = [all_fine_images[i] for i in random_images_index]
        random_fixation_image_dic = [all_fixation_images[i] for i in random_images_index]

        return  random_coarse_image_dic, random_fine_image_dic, random_fixation_image_dic

    def run(self):

        'get the train data'
        all_images = 15491
        batch_size = 50
        epochs = 5

        all_iteration = int(all_images / batch_size)
        print '>>>>>>>>>>>>>>>>>>>>>: all_iteration ', all_iteration

        start_time_0 = time.time()
        for i_epochs in range(epochs):
            random_coarse_image_dic, random_fine_image_dic, random_fixation_image_dic = self.get_image_list_of_jiang('/home/ml/OpenSALICON/Data/LEDOV-Salicon/')
            for i_iterate in range(all_iteration):
                time_iterate_0 = time.time()
                'get data in each batch'
                i_start_frame = i_iterate * batch_size
                coarse_imgs, fine_imgs, fix_imgs = self.get_one_batch_data(coarse_image_dic = random_coarse_image_dic,
                                                                           fine_image_dic =  random_fine_image_dic,
                                                                           fixation_image_dic = random_fixation_image_dic,
                                                                           start_frame = i_start_frame,
                                                                           batch_size = batch_size)

                print '>>>>>>>>>>>>>>>>>>> load data cost ', str(time.time() - time_iterate_0)
                'train the model'
                i_train_time = time.time()

                self.train(coarse_imgs, fine_imgs, fix_imgs, i_batch = i_iterate, batch_size = batch_size)

                print '>>>>>>>>>>>>>>>> train_one_batch time: ', str(time.time() - i_train_time)
                print '>>>>>>>>>>>>total time cost: ', str(time.time() - start_time_0)


if __name__ == '__main__':

    test_finetune =  Fine_tune()
    # test_finetune.run()
    test_finetune.run()
