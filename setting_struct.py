class setting_option:
    def __init__(self,
                 name,
                 stage,
                 datamode,
                 data_list,
                 checkpoint,
                 save_count,
                 shuffle,

                 batch_size=16,
                 lr=1e-4,
                 dataroot="data",
                 tensorboard_dir="tensorboard",
                 result_dir="result",
                 checkpoint_dir="checkpoints",
                 display_count=1,
                 fine_width=192,
                 fine_height=256,
                 radius=5,
                 grid_size=5,
                 keep_step=100000,
                 decay_step=100000,
                 ):

        self.batch_size = batch_size
        self.lr = lr

        self.name = name
        # gmm_train_new gmm_traintest_new
        # tom_train_new tom_test_new

        self.stage = stage
        # GMM TOM
        self.datamode = datamode
        # train test

        self.dataroot = dataroot
        self.data_list = data_list
        # train_pairs.txt test_pairs.txt
        self.tensorboard_dir = tensorboard_dir
        self.result_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = checkpoint
        # checkpoints/gmm_train_new/gmm_final.pth
        # checkpoints/tom_train_new/tom_final.pth

        self.shuffle = shuffle
        self.display_count = display_count
        self.fine_width = fine_width
        self.fine_height = fine_height
        self.radius = radius
        self.grid_size = grid_size
        self.save_count = save_count
        # 5000 100
        self.keep_step = keep_step
        self.decay_step = decay_step


'''

python train.py --name gmm_train_new --stage GMM --workers 4 --save_count 5000 --shuffle
python test.py --name gmm_traintest_new --stage GMM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/gmm_train_new/gmm_final.pth
python train.py --name tom_train_new --stage TOM --workers 4 --save_count 5000 --shuffle 
python test.py --name tom_test_new --stage TOM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/tom_train_new/tom_final.pth

'''

# if __name__ == "__main__":
#     opt = setting_option(
#         name="tom_train_new",
#         stage="TOM",
#         datamode="train",
#         data_list="train_pairs.txt",
#         checkpoint="",
#         save_count=5000,
#         shuffle=True
#     )
#     print(opt.decay_step)