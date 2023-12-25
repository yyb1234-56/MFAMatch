'''这个实验是来观察训练出的多源MatchNet剩余200张测试集上匹配的效果
去除误匹配点的方法就暂时使用三点向量法和中值法，模板图512大小，参考图800大小，看看效果'''
import torch.nn.functional as F
import numpy as np
import torch
import xlwt
import operator
import cv2
from MFAMatch_model import MFAMatch_Net
import os
from torchvision import transforms
from super_utils import Fbox
import random
import xlrd
import time
from datetime import datetime


def read_excel(path,row):
    # 打开文件
    workbook = xlrd.open_workbook(path)
    # 获取所有sheet
    # print (workbook.sheet_names()) # [u'sheet1', u'sheet2']
    # 根据sheet索引或者名称获取sheet内容
    sheet1 = workbook.sheet_by_name('data')
    row_value = sheet1.row_values(int(row))

    return  row_value

def socres_calculate1(path):
    correct_num_10 = []
    correct_num_6 = []
    correct_num_3 = []
    correct_num_1 = []
    book = xlrd.open_workbook(path)# 打开Excel文件
    # 获取第一个工作表
    sheet = book.sheet_by_index(0)
    # 获取表格的行数
    num_rows = sheet.nrows-1
    for row in range(1,num_rows-1):#第一行是列名
        data = read_excel(path, row)
        if data[2] <= 10:
            correct_num_10.append(data[2])
        if data[2] <= 6:
            correct_num_6.append(data[2])
        if data[2] <= 3:
            correct_num_3.append(data[2])
        if data[2] <= 1:
            correct_num_1.append(data[2])

    result = '实验共{}组，误差≤10：{:.3f}，均值：{:.3f}， 误差≤6：{:.3f}，均值：{:.3f}， 误差≤3：{:.3f}，均值：{:.3f}， 误差≤1：{:.3f}，均值：{:.3f}， '.format(num_rows, len(correct_num_10)/num_rows, np.mean(correct_num_10), len(correct_num_6)/num_rows,
                                                                           np.mean(correct_num_6), len(correct_num_3)/num_rows, np.mean(correct_num_3),  len(correct_num_1)/num_rows, np.mean(correct_num_1))
    print(result)
    return result
def find_files(_data_dir, _image_ext):
    """Return a list with the file names of the images containing the patches
    """
    files = []
    # find those files with the specified extension
    for file_dir in os.listdir(_data_dir):
        if file_dir.endswith(_image_ext):
            files.append(os.path.join(_data_dir, file_dir))
    return sorted(
        files)  # sort files in ascend order to keep relations   the serial numbers are complex whether sorted or not eg. 1-10-100-101


def read_coordinate(dir):
    with open(dir, 'r') as f:
        for line in f:
            line_split = line.split()
            coordinate_x = int(line_split[0])
            coordinate_y = int(line_split[1])
            if (coordinate_x >= 288 or coordinate_y >= 288):
                print('x:', coordinate_x, '  y:', coordinate_y, ' label_path:', dir)
                #raise RuntimeError('the value of x-y is wrong')
    return coordinate_x, coordinate_y
def feature_pyramid_calculate(transform_f, img, net, DEVICE):
    img_temp = transform_f(img)
    img_feature_f1 = net.f1(img_temp.unsqueeze(0).to(DEVICE))
    img_feature_f2 = net.f2(img_feature_f1)
    img_feature_f3 = net.f3(img_feature_f2)
    img_feature_f4 = net.f4(img_feature_f3)
    img_feature_f5 = net.f5(img_feature_f4)
    img_feature_pyramid = [img_feature_f2, img_feature_f3, img_feature_f4, img_feature_f5]
    return img_feature_pyramid

def feature_extract(feature_pyramid, p, q,_pix):
    resample_ratio_list = [4,8,16,32]
    feature_size_list = [int(_pix/ratio) for ratio in resample_ratio_list]
    feature_coord_list = [[p//ratio, q//ratio] for ratio in resample_ratio_list]
    feature_list = []
    for i in range(len(resample_ratio_list)):
        x, y = feature_coord_list[i][0],  feature_coord_list[i][1]
        feature_list.append(feature_pyramid[i][:,:, x:x + feature_size_list[i], y:y + feature_size_list[i]])
    img_feature_f2, img_feature_f3,  img_feature_f4, img_feature_f5 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]
    img_up2 = net.conv2(img_feature_f2)
    img_up3 = net.up3(img_feature_f3)
    img_up4 = net.up4(img_feature_f4)
    img_up5 = net.up5(img_feature_f5)
    img_feature_mix1 = torch.cat((img_up2, img_up3, img_up4, img_up5), dim=1)
    img_feature_mix2 = net.conv5(net.conv4(net.conv3(img_feature_mix1)))
    img_feature_mix1 = img_feature_mix1.view(img_feature_mix1.size(0), -1)
    img_feature_mix2 = img_feature_mix2.view(img_feature_mix2.size(0), -1)
    img_feature_mix1_norm = F.normalize(img_feature_mix1, dim=1)
    img_feature_mix2_norm = F.normalize(img_feature_mix2, dim=1)
    return img_feature_mix1_norm, img_feature_mix2_norm

def model_calculate(transform_f, img, net, DEVICE):
    img_temp = transform_f(img)
    img_feature_f1 = net.f1(img_temp.unsqueeze(0).to(DEVICE))
    img_feature_f2 = net.f2(img_feature_f1)
    img_feature_f3 = net.f3(img_feature_f2)
    img_feature_f4 = net.f4(img_feature_f3)
    img_feature_f5 = net.f5(img_feature_f4)
    img_up2 = net.conv2(img_feature_f2)
    img_up3 = net.up3(img_feature_f3)
    img_up4 = net.up4(img_feature_f4)
    img_up5 = net.up5(img_feature_f5)
    img_feature_mix1 = torch.cat((img_up2, img_up3, img_up4, img_up5), dim=1)
    img_feature_mix2 = net.conv5(net.conv4(net.conv3(img_feature_mix1)))
    img_feature_mix1 = img_feature_mix1.view(img_feature_mix1.size(0), -1)
    img_feature_mix2 = img_feature_mix2.view(img_feature_mix2.size(0), -1)
    img_feature_mix1_norm = F.normalize(img_feature_mix1, dim=1)
    img_feature_mix2_norm = F.normalize(img_feature_mix2, dim=1)
    return img_feature_mix1_norm, img_feature_mix2_norm

def Weapon(_pix, envi, stride, stride1,  opt_dir, sar_dir, save_dir, model,  DEVICE, savepath_xls1 , weight):

    image_ext = 'png' #tif

    transform_test_opt = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.35876667,), std=(0.3196282,))])  # 归一化
    transform_test_sar = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.24362623,), std=(0.1922179,))])  # 归一化

    fine_error = []
    corase_error = []
    fine_correct_num = 0
    corase_correct_num = 0
    sum = 0
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('data', cell_overwrite_ok=True)
    col1 = ('path', 'coarse_error', 'fine_error')
    for i in range(0, 3):
        sheet.write(0, i, col1[i])
    list_files1 = find_files(opt_dir, image_ext)
    list_files2 = find_files(sar_dir, image_ext)
    col = 1
    random.seed(1234)  # 这个随机种子非常非常重要！！！
    start_time = time.time()
    for count in range(len(list_files1)):
        # 数据准备
        fpath1 = list_files1[count]
        fpath2 = list_files2[count]

        img_opt = cv2.imread(fpath1, 0)  # opt
        opt_size = img_opt.shape

        img_sar_origin = cv2.imread(fpath2, 0)
        y_label, x_label = random.randint(0, img_opt.shape[0]-_pix), random.randint(0, img_opt.shape[0]-_pix)
        '''一个列表中每两行是一次数据的结果，每num*2行是一张图片的num次随机结果'''
        print('第{}张'.format(count+1))
        img_sar = img_sar_origin[x_label:x_label + _pix, y_label: y_label+_pix]

        _save_dir1 = save_dir + '/' + str(count)
        if os.path.exists(_save_dir1) != 1:
            os.makedirs(_save_dir1)

        path = _save_dir1 + '/opt.jpg'
        path1 = _save_dir1 + '/opt1.jpg'
        sar_path = _save_dir1 + '/sar_patch.jpg'
        cv2.imwrite(path, img_opt)
        cv2.imwrite(path1, img_opt)
        cv2.imwrite(sar_path, img_sar)

        coordinate_list = []
        outputs_list = []
        sar_feature_norm = model_calculate(transform_test_sar, img_sar, net, DEVICE)
        opt_feature_pyramid = feature_pyramid_calculate(transform_test_opt, img_opt, net, DEVICE)
        with torch.no_grad():
            model.eval()
            for p in range(0, opt_size[0] - _pix, stride):
                for q in range(0, opt_size[0] - _pix, stride):
                    opt_feature_norm = feature_extract(opt_feature_pyramid, p, q, _pix)
                    # opt_feature_norm = coarse_model_calculate(transform_test_opt, img_opt[p:(p + _pix), q:(q + _pix)], net, DEVICE)
                    outputs = weight*torch.nn.functional.cosine_similarity(sar_feature_norm[0], opt_feature_norm[0], dim=1) + (1-weight)*torch.nn.functional.cosine_similarity(sar_feature_norm[1], opt_feature_norm[1], dim=1)
                    outputs_list.append(outputs.to('cpu').numpy())
                    coordinate_list.append([p, q])

        temp = zip(outputs_list, coordinate_list)
        sorted_outputs = sorted(temp, key=operator.itemgetter(0), reverse=True)
        xy_pair = sorted_outputs[0][1]
        Fbox.draw_ncircle_new(y_label, x_label, xy_pair, path)
        # 画出16格的中点及三角向量法预测点
        img = cv2.imread(path)
        true_center_x = x_label
        true_center_y = y_label
        #print('true_center_x:', true_center_x, ' ', 'true_center_y:', true_center_y)
        print('粗匹配结果：')
        cv2.rectangle(img, (true_center_y - 4, true_center_x - 4), (true_center_y + 4, true_center_x + 4),
                      (255, 0, 255), -1)

        # pred_sjx_x, pred_sjx_y, min_diff_vector = Fbox.ransac16_new(dict_true, dict_pred, dict_center_distance, 3)
        pred_sjx_x, pred_sjx_y = xy_pair[0], xy_pair[1]
        cv2.rectangle(img, (int(pred_sjx_y) - 4, int(pred_sjx_x) - 4), (int(pred_sjx_y) + 4, int(pred_sjx_x) + 4), (255, 153, 204), -1)
        cv2.imwrite(path, img)
        print('xerror:', str(abs(true_center_x - pred_sjx_x)),'yerror:', str(abs(true_center_y - pred_sjx_y)))
        corase_error_temp = ((true_center_x - pred_sjx_x) ** 2 + (true_center_y - pred_sjx_y) ** 2) ** 0.5
        if corase_error_temp < 10:
            corase_error.append(corase_error_temp)
            corase_correct_num += 1

        del xy_pair, sar_feature_norm, opt_feature_norm

        coordinate_list = []
        outputs_list = []
        with torch.no_grad():
            # sar_feature_norm = fine_model_calculate(transform_test_sar,img_sar, net, DEVICE)
            sar_feature_norm = model_calculate(transform_test_sar, img_sar, net, DEVICE)
            for p in range(max([0,pred_sjx_x-envi]), min([img_opt.shape[0]-_pix, pred_sjx_x+envi]), stride1):
                for q in range(max([0,pred_sjx_y-envi]), min([img_opt.shape[0]-_pix, pred_sjx_y+envi]), stride1):
                    # opt_feature_norm = fine_model_calculate(transform_test_opt,img_opt[p:(p + _pix), q:(q + _pix)], net, DEVICE)
                    opt_feature_norm = model_calculate(transform_test_opt, img_opt[p:(p + _pix), q:(q + _pix)], net, DEVICE)
                    # outputs = torch.nn.functional.cosine_similarity(sar_feature_norm, opt_feature_norm, dim=1)
                    # outputs_list.append(outputs.to('cpu').numpy())
                    outputs = weight * torch.nn.functional.cosine_similarity(sar_feature_norm[0], opt_feature_norm[0],
                                                                          dim=1) + (1-weight) * torch.nn.functional.cosine_similarity(
                        sar_feature_norm[1], opt_feature_norm[1], dim=1)
                    outputs_list.append(outputs.to('cpu').numpy())
                    coordinate_list.append([p, q])
        temp = zip(outputs_list, coordinate_list)
        sorted_outputs = sorted(temp, key=operator.itemgetter(0), reverse=True)
        xy_pair = sorted_outputs[0][1]
        Fbox.draw_ncircle_new(y_label, x_label, xy_pair, path1)
        # 画出16格的中点及三角向量法预测点
        img = cv2.imread(path1)
        # print('true_center_x:', true_center_x, ' ', 'true_center_y:', true_center_y)
        print('精匹配结果：')
        cv2.rectangle(img, (true_center_y - 4, true_center_x - 4), (true_center_y + 4, true_center_x + 4),
                      (255, 0, 255), -1)

        pred_jpp_x, pred_jpp_y = xy_pair[0], xy_pair[1]
        cv2.rectangle(img, (int(pred_jpp_y) - 4, int(pred_jpp_x) - 4), (int(pred_jpp_y) + 4, int(pred_jpp_x) + 4),
                      (255, 153, 204), -1)
        cv2.imwrite(path, img)
        print('xerror:', str(abs(true_center_x - pred_jpp_x)), 'yerror:', str(abs(true_center_y - pred_jpp_y)))
        fine_error_temp = ((true_center_x - pred_jpp_x) ** 2 + (true_center_y - pred_jpp_y) ** 2) ** 0.5
        if fine_error_temp <= 3:
            fine_error.append(fine_error_temp)
            fine_correct_num += 1
        sheet.write(col, 0, fpath1)
        sheet.write(col, 1, corase_error_temp)
        sheet.write(col, 2, fine_error_temp)
        sum = sum + 1
        col = col + 1
        book.save(savepath_xls1)
    # 训练结束后获取当前时间
    end_time = time.time()

    # 计算训练时间（单位：小时）
    training_time = end_time - start_time
    print('测试时间：{:.2f} s'.format(training_time ))
    print('粗匹配 valid_num:', sum, 'correct_num:', corase_correct_num, 'rate:', corase_correct_num / sum,'n_points_avg:',
          str(np.mean(corase_error)), 'n_points_std:', str(np.std(corase_error)))
    print('精匹配 valid_num:', sum, 'correct_num:', fine_correct_num, 'rate:', fine_correct_num / sum,
          'n_points_avg:',
          str(np.mean(fine_error)), 'n_points_std:', str(np.std(fine_error)))


if __name__ == '__main__':

    _pix = 128
    envi = 8
    stride = 4
    stride1 = 1
    ########################随机种子是1234######################
    #模型选择
    imgPath1 = 'D:/YYB_code_data/origin_data/OSdataset/512/train_opt'
    imgPath2 = 'D:/YYB_code_data/origin_data/OSdataset/512/train_sar'
    label_path = 'None'
    weight = 0.7
    model_path = '../MFAMatch/pretrained_model/model_weight0.7.pth'
    save_dir = './test_result/template_match_weight0.7'
    savepath_xls1 = './test_result/template_match_weight0.7.xlsx'

    experiment_description = '实验目的：' + '检验多层特征叠加以及逐步融合之后的特征共同加权的网络(后者权重'+str(weight) + '）在osdataset上的效果，训练集大小缩小到1/2,验证集目前2.91/74.54，初始学习率0.01，epoch =16，这里做512找128,1000组测试' + '\n实验时间' + str(datetime.now()) +  \
                             '\n实验数据集：' + imgPath1 + '\n模型： ' + model_path + '\n保存路径： ' + save_dir + '\n' + savepath_xls1

    model = MFAMatch_Net(use_bn=False, use_lrn=False, input_channels=1, bottleneck_size=512, fullyconect_size=512, num_class=2)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint_rough = torch.load(model_path)
    model.load_state_dict(checkpoint_rough)
    net = model.to(DEVICE)

    Weapon(_pix, envi, stride, stride1, imgPath1, imgPath2,  save_dir, net,  DEVICE, savepath_xls1, weight)
    print(experiment_description)
    result = socres_calculate1(savepath_xls1)
    # 可以直接输数据进行精匹配
