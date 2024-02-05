import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--main_dir", type=str,default='/home/mojjaf/Pix2Pix_CardiacPET/Pix2Pix_3D_v05',
                        help="project directory")
    
    parser.add_argument("--data_dir", type=str,default='/home/mojjaf/Pix2Pix_CardiacPET/data/',
                        help="directory for all images")
    
    parser.add_argument("--image_size", type=int, default=128,
                        help="training patch size")
    
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size")
    
    parser.add_argument("--input_channel", type=int, default=48,
                        help="number of input channels for 2D=1 or 2.5D=3 or 3D=48")
        
    parser.add_argument("--output_channel", type=int, default=48,
                        help="number of output channels for 2D or 2.5D or 3D")
   
    parser.add_argument("--nb_epochs", type=int, default=120,
                        help="number of epochs")
    
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="learning rate")
    
    parser.add_argument("--steps", type=int, default=2000,
                        help="steps per epoch")#was 650000
    
    parser.add_argument("--loss", type=str, default=None,
                        help="loss; mean_squared_error', 'mae', or 'l0' is expected")
    
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
   
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    return args 
