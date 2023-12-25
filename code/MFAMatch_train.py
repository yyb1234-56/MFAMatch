import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from torch.optim.lr_scheduler import StepLR
from MFAMatch_model import MFAMatch_Net
from Data_Transform import myTransform_poc_8
from super_utils import Weighted_Quintuplet_Soft_Loss
from super_utils import Fbox
import xlwt
import time

torch.manual_seed(3456)
torch.backends.cudnn.deterministic = True

class Config():
    BATCH_SIZE = 16
    EPOCHS = 60
    LEARNING_RATE = 0.01#0.01
    LR_GAMMA = 0.5
    INTERVAL = 3  # 间隔
    INPUT_SIZE = 64
    INPUT_CHANNELS = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(train_file_paths, model, batch_size, device, optimizer, epoch, n, criterion):
    train_rmse_max6_list = []
    train_rmse_max7_list = []
    train_rmse_min7_list = []
    train_loss_list = []

    for train_index, file_path in enumerate(train_file_paths):
        train_datasets_temp = myTransform_poc_8(file_path, train=True)
        train_dataloader = torch.utils.data.DataLoader(train_datasets_temp, batch_size, pin_memory=True, shuffle=True, num_workers=4)
        train_rmse_max6, train_rmse_max7, train_rmse_min7, train_loss = Fbox.poc_train_4_input(model, device, train_dataloader, optimizer, epoch, train_index, n, criterion)

        # 记录每个循环的结果
        train_rmse_max6_list.append(train_rmse_max6)
        train_rmse_max7_list.append(train_rmse_max7)
        train_rmse_min7_list.append(train_rmse_min7)
        train_loss_list.append(train_loss)

    # 计算均值
    avg_train_rmse_max6 = sum(train_rmse_max6_list) / len(train_rmse_max6_list)
    avg_train_rmse_max7 = sum(train_rmse_max7_list) / len(train_rmse_max7_list)
    avg_train_rmse_min7 = sum(train_rmse_min7_list) / len(train_rmse_min7_list)
    avg_train_loss = sum(train_loss_list) / len(train_loss_list)
    del train_datasets_temp, train_dataloader
    # 构建结果字符串
    result_str = "Epoch: {} 平均训练结果： Train RMSE Max6: {:.2f}, Train RMSE Max7: {:.2f}, Train RMSE Min7: {:.2f}, Train Loss: {:.2f}".format(epoch,
        avg_train_rmse_max6, avg_train_rmse_max7, avg_train_rmse_min7, avg_train_loss)
    print(result_str)
    return avg_train_rmse_max6, avg_train_rmse_max7, avg_train_rmse_min7, avg_train_loss
def test_loop(test_file_paths, model, batch_size, device, epoch, n, criterion):
    test_rmse_max6_list = []
    test_rmse_max7_list = []
    test_rmse_min7_list = []
    test_loss_list = []

    for test_index, file_path in enumerate(test_file_paths):
        test_datasets_temp = myTransform_poc_8(file_path)
        test_dataloader = torch.utils.data.DataLoader(test_datasets_temp, batch_size, pin_memory=True, shuffle=True, num_workers=4)
        test_rmse_max6, test_rmse_max7, test_rmse_min7, test_loss = Fbox.poc_test_4_input(model, device, test_dataloader, epoch, test_index, n, criterion)

        # 记录每个循环的结果
        test_rmse_max6_list.append(test_rmse_max6)
        test_rmse_max7_list.append(test_rmse_max7)
        test_rmse_min7_list.append(test_rmse_min7)
        test_loss_list.append(test_loss)
    # 计算均值
    avg_test_rmse_max6 = sum(test_rmse_max6_list) / len(test_rmse_max6_list)
    avg_test_rmse_max7 = sum(test_rmse_max7_list) / len(test_rmse_max7_list)
    avg_test_rmse_min7 = sum(test_rmse_min7_list) / len(test_rmse_min7_list)
    avg_test_loss = sum(test_loss_list) / len(test_loss_list)
    del test_datasets_temp, test_dataloader
    # 构建结果字符串
    result_str = "Epoch: {} 平均测试结果： test RMSE Max6: {:.2f}, test RMSE Max7: {:.2f}, test RMSE Min7: {:.2f}, test Loss: {:.2f}".format(epoch,
        avg_test_rmse_max6, avg_test_rmse_max7, avg_test_rmse_min7, avg_test_loss)
    print(result_str)
    return avg_test_rmse_max6, avg_test_rmse_max7, avg_test_rmse_min7, avg_test_loss

if __name__ == '__main__':

    net1 = MFAMatch_Net(use_bn=False, use_lrn=False, input_channels=1, bottleneck_size=512, fullyconect_size=512, num_class=0)
    model = net1.to(Config.DEVICE)

    pre_rmse = 100
    discription = 'MatchNet_Multiscale_upsample_v4_mix_0.5_loss_v3_lr0.01_weight0.7_11-1'
    dataset_save_dir = "D:/YYB_code_data/dataset_new/yyb_report_os/Multilayer/"
    n = 20
    Path_model_save = 'C:/Users/CV/Desktop/Paper/model_save/'+discription+'/'
    fig_path = 'C:/Users/CV/Desktop/Paper/jpg_xls_save/'+discription+'_B{}L{}.png'.format(Config.BATCH_SIZE, Config.LEARNING_RATE)
    savepath_xls = 'C:/Users/CV/Desktop/Paper/jpg_xls_save/'+discription+'_B{}L{}.xls'.format(Config.BATCH_SIZE, Config.LEARNING_RATE)
    Fbox.makedir(Path_model_save)
    criterion = Weighted_Quintuplet_Soft_Loss(feature_weight =0.7)
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=Config.INTERVAL, gamma=Config.LR_GAMMA)
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)

    train_loss_list = []
    train_rmse_max6_list = []
    test_rmse_max6_list = []
    train_rmse_max7_list = []
    test_rmse_max7_list = []
    train_rmse_min7_list = []
    test_rmse_min7_list = []
    test_loss_list = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    print('Training!')
    sheet = book.add_sheet('data', cell_overwrite_ok=True)
    col = ('train_loss', 'test_loss_list', 'train_rmse_max6', 'test_rmse_max6', 'train_rmse_max7', 'test_rmse_max7', 'train_rmse_min7', 'test_rmse_min7')
    for i in range(0, 8):
        sheet.write(0, i, col[i])

    col1 = 1
    # 遍历所有数据包
    for epoch in range(1,Config.EPOCHS + 1):
        start_time = time.time()
        print('第{}代 '.format(epoch))
        # 加载训练数据

        train_file_paths = [f"{dataset_save_dir}/train20_0.5part{part}.pt" for part in range(n)]
        train_rmse_max6, train_rmse_max7, train_rmse_min7, train_loss = train_loop(train_file_paths, model, Config.BATCH_SIZE, Config.DEVICE, optimizer, epoch, n, criterion)

        # 构建文件路径列表
        test_file_paths = [f"{dataset_save_dir}/test20_0.5part{part}.pt" for part in range(n)]
        test_rmse_max6, test_rmse_max7, test_rmse_min7, test_loss = test_loop(test_file_paths, model, Config.BATCH_SIZE, Config.DEVICE, epoch, n, criterion)
        # 释放内存

        # del test_dataloader_list

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_rmse_max6_list.append(train_rmse_max6)
        test_rmse_max6_list.append(test_rmse_max6)
        train_rmse_max7_list.append(train_rmse_max7)
        test_rmse_max7_list.append(test_rmse_max7)
        train_rmse_min7_list.append(train_rmse_min7)
        test_rmse_min7_list.append(test_rmse_min7)
        scheduler.step()
        data = [train_loss, test_loss, train_rmse_max6, test_rmse_max6, train_rmse_max7, test_rmse_max7, train_rmse_min7, test_rmse_min7]
        for j in range(0, 8):
            sheet.write(col1, j, data[j])
        col1 = col1 + 1
        book.save(savepath_xls)
        if test_rmse_max6 < pre_rmse:
            pre_acc = test_rmse_max6
        torch.save(model.state_dict(), Path_model_save + str(epoch) + '_model.pth')
        # 训练结束后获取当前时间
        end_time = time.time()

        # 计算训练时间（单位：小时）
        training_time = end_time - start_time
        print('训练时间：{:.3f} 分钟'.format(training_time/60))



    print('best_rmse:',pre_rmse)
    end.record()
    torch.cuda.synchronize()

    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    plt.ion()
    plt.xlabel('Epoch ')
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.plot(train_rmse_max6_list)
    plt.plot(test_rmse_max6_list)
    plt.plot(train_rmse_max7_list)
    plt.plot(test_rmse_max7_list)
    plt.plot(train_rmse_min7_list)
    plt.plot(test_rmse_min7_list)
    plt.legend(['train_loss', 'test_loss_list', 'train_rmse_max6', 'test_rmse_max6', 'train_rmse_max7', 'test_rmse_max7', 'train_rmse_min7', 'test_rmse_min7'])
    plt.title('lr = {} step 10 batch = {} acc={:.3f} P:N = 1:3 dataset_'.format(Config.LEARNING_RATE, Config.BATCH_SIZE, test_rmse_max7_list[-1]) )
    plt.savefig(fig_path)
    plt.show()
    plt.close()
