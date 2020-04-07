####--------------------Get caffe model from pytorch refinedet_res10---------------------------#############
####    1_read_weights.py ->First read the pytorch model (*.pth)  and save the weight dict to (*.npy)
####   2 prepare the caffe *.prototxt 
####      copy the npy layer weight to correspond caffe layer 
####      last produce the caffemodel (*.caffemodel)
####------------------------------------------------------------------------------------------##############
