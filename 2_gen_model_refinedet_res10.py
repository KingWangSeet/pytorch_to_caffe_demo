
####--------------------Get caffe model from pytorch refinedet_res10---------------------------#############
####    1_read_weights.py ->First read the pytorch model (*.pth)  and save the weight dict to (*.npy)
####   2 prepare the caffe *.prototxt 
####      copy the npy layer weight to correspond caffe layer 
####      last produce the caffemodel (*.caffemodel)
####------------------------------------------------------------------------------------------##############

import sys
sys.path.insert(0, 'caffe/python')
import sys
import os
import caffe
import numpy as np

# checkpoint_path = "refinedet_res10_2cls.npy"  #input from the 1_read_weights.py
# checkpoint_path = "refinedet_res10_2cls_nofocaloss.npy"  #input from the 1_read_weights.py
checkpoint_path = "refinedet_res10_2cls_focaloss_two_T.npy"  #input from the 1_read_weights.py

cf_prototxt = "refinedet_res10.prototxt" #input the refindedet res10cfg

cf_model = "refinedet_res10_2cls_focaloss_T.caffemodel" #output caffe_model

var_to_weights_map = np.load(checkpoint_path)[()]

#unused
def extract_from_conv_block(net, var_to_weights_map, stage, block):
    conv_name_base = 'layer'+str(stage-1)+'.'+str(block)
    bn_name_base = 'layer' + str(stage-1)+'.'+str(block)

    caffe_name_base = 'res' + str(stage) #+ block + '_branch'
    if int(block) == 0:
    # a     
        caffe_name_base = caffe_name_base + chr(int(block)+97) + '_branch'
    else:
        caffe_name_base = caffe_name_base + 'b'+str(block) + '_branch'
    name = [[conv_name_base+'.conv1', bn_name_base+'.bn1', caffe_name_base + '2a'],
            [conv_name_base+'.conv2', bn_name_base+'.bn2', caffe_name_base + '2b']]
    
    for conv_name, bn_name, caffe_name in name:         
        # conv
        weights = var_to_weights_map[conv_name+'.weight']
        net.params[caffe_name][0].data.flat = weights.flat
        # bn
        mean = var_to_weights_map[bn_name+'.running_mean']
        net.params[caffe_name+'/bn'][0].data.flat = mean.flat
        variance = var_to_weights_map[bn_name+'.running_var']
        net.params[caffe_name+'/bn'][1].data.flat = variance.flat
        net.params[caffe_name+'/bn'][2].data.flat = np.array([1]).flat
        gamma = var_to_weights_map[bn_name+'.weight']
        net.params[caffe_name+'/scale'][0].data.flat = gamma.flat    
        beta = var_to_weights_map[bn_name+'.bias']
        net.params[caffe_name+'/scale'][1].data.flat = beta.flat 

    if int(block)==0 and stage>2:
    # a  
        # shortcut  
        caffe_short_cut_name = caffe_name_base + '1'
        conv_name = conv_name_base + '.downsample'
        bn_name = conv_name_base + '.downsample'
        # conv
        weights = var_to_weights_map[conv_name+'.0.weight']
        net.params[caffe_short_cut_name][0].data.flat = weights.flat
        # bn
        mean = var_to_weights_map[bn_name+'.1.running_mean']
        net.params[caffe_short_cut_name+'/bn'][0].data.flat = mean.flat
        variance = var_to_weights_map[bn_name+'.1.running_var']
        net.params[caffe_short_cut_name+'/bn'][1].data.flat = variance.flat
        net.params[caffe_short_cut_name+'/bn'][2].data.flat = np.array([1]).flat
        gamma = var_to_weights_map[bn_name+'.1.weight']
        net.params[caffe_short_cut_name+'/scale'][0].data.flat = gamma.flat    
        beta = var_to_weights_map[bn_name+'.1.bias']
        net.params[caffe_short_cut_name+'/scale'][1].data.flat = beta.flat   

#SEbasicBlock
def extract_from_seres10_sebasicblock(net, var_to_weights_map, stage):
    block =0
    #var_to_weights_map key
    conv_name_base = 'layer'+str(stage-1)+'.'+str(block) #layer1.0
    bn_name_base = 'layer' + str(stage-1)+'.'+str(block)
    
    caffe_name_base = 'conv' + str(stage)+'_1' #conv2-1
    caffe_name_base_x1 = 'conv' + str(stage)+'_1/x1' #conv2_1/x1
    caffe_name_base_x2 = 'conv' + str(stage)+'_1/x2' #conv2_1/x2
       
    #SE basicblok conv x1 x2
    name = [[conv_name_base+'.conv1', bn_name_base+'.bn1', caffe_name_base_x1], #layer1.0.conv1  layer1.0.bn1 conv2_1/x1
            [conv_name_base+'.conv2', bn_name_base+'.bn2', caffe_name_base_x2]]#layer1.0.conv2 layer1.0.bn2 onv2_1/x2
    
    for conv_name, bn_name, caffe_name in name:         
        # conv
#         print(conv_name, bn_name, caffe_name) #caffe_name -> conv2_1/x1
        weights = var_to_weights_map[conv_name+'.weight']
#         print(weights)
        net.params[caffe_name][0].data.flat = weights.flat #ok
#         print(net.params[caffe_name][0].data)
#         print('differ between layer1.0.conv1.weight')
#         print((weights-net.params[caffe_name][0].data).mean())#0
#         exit()
        # bn
        mean = var_to_weights_map[bn_name+'.running_mean']
#         print(mean)
        net.params[caffe_name+'/bn'][0].data.flat = mean.flat #ok

        variance = var_to_weights_map[bn_name+'.running_var']
#         print(variance)
        net.params[caffe_name+'/bn'][1].data.flat = variance.flat
        net.params[caffe_name+'/bn'][2].data.flat = np.array([1]).flat
        
        gamma = var_to_weights_map[bn_name+'.weight']
#         print(gamma)
        net.params[caffe_name+'/scale'][0].data.flat = gamma.flat  #ok
        
        beta = var_to_weights_map[bn_name+'.bias']
#         print(beta)
        net.params[caffe_name+'/scale'][1].data.flat = beta.flat #ok
#         print(net.params[caffe_name+'/scale'][1].data)
#         print((beta-net.params[caffe_name+'/scale'][1].data).mean())#0
#         exit()
#     exit()
    #SE basicblok SE layer
    #layer1.0.se.fc.0.weight  
    #layer1.0.se.fc.0.bias
    #layer1.0.se.fc.2.weight
    #layer1.0.se.fc.2.bias
    
    conv_se_fc_name_0 = conv_name_base + '.se.fc.0' #layer1.0.se.fc.0
    conv_se_fc_name_2 = conv_name_base + '.se.fc.2' #layer1.0.se.fc.2
    
    caffe_se_fc_name_0 = 'fc'+str(stage)+'_1/sqz' #fc2_1/sqz
    caffe_se_fc_name_2 = 'fc'+str(stage)+'_1/exc' #fc2_1/exc
    
    se_fc_0_weight = var_to_weights_map[conv_se_fc_name_0+'.weight']
    se_fc_0_bias  = var_to_weights_map[conv_se_fc_name_0+'.bias']
#     print('se_fc_0_weight')
#     print(se_fc_0_weight.shape)
#     print(caffe_se_fc_name_0)
#     print(net.params[caffe_se_fc_name_0][0].data.shape)
#     exit()
    se_fc_2_weight = var_to_weights_map[conv_se_fc_name_2+'.weight']
    se_fc_2_bias  = var_to_weights_map[conv_se_fc_name_2+'.bias']
    
    
    net.params[caffe_se_fc_name_0][0].data.flat = se_fc_0_weight.flat
    net.params[caffe_se_fc_name_0][1].data.flat = se_fc_0_bias.flat
    
    net.params[caffe_se_fc_name_2][0].data.flat = se_fc_2_weight.flat
    net.params[caffe_se_fc_name_2][1].data.flat = se_fc_2_bias.flat
    
    
    #downsampling layer2->conv3_1 layer3->conv4_1 layer->4conv5_1
    #layer2.0.downsample.0.weight 
    #layer2.0.downsample.1.weight
    #layer2.0.downsample.1.bias
    #layer2.0.downsample.1.running_mean
    #layer2.0.downsample.1.running_var
    
    if stage>2: 
        caffe_short_cut_name = caffe_name_base + '/prj' #conv3_1/prj
        
        conv_name = conv_name_base + '.downsample' #layer2.0.downsample
        bn_name = bn_name_base + '.downsample'
        # conv
        weights = var_to_weights_map[conv_name+'.0.weight']
        net.params[caffe_short_cut_name][0].data.flat = weights.flat
        # bn
        mean = var_to_weights_map[bn_name+'.1.running_mean']
        net.params[caffe_short_cut_name+'/bn'][0].data.flat = mean.flat
        variance = var_to_weights_map[bn_name+'.1.running_var']
        net.params[caffe_short_cut_name+'/bn'][1].data.flat = variance.flat
        net.params[caffe_short_cut_name+'/bn'][2].data.flat = np.array([1]).flat
        gamma = var_to_weights_map[bn_name+'.1.weight']
        net.params[caffe_short_cut_name+'/scale'][0].data.flat = gamma.flat    
        beta = var_to_weights_map[bn_name+'.1.bias']
        net.params[caffe_short_cut_name+'/scale'][1].data.flat = beta.flat   
     
    #print('fininsing loading weight from ' +conv_name_base+ ' to  '+caffe_name_base)
        
def torch2caffe(var_to_weights_map, net):    
    # Resnet
    # Stage 1
    # conv1/weights
    weights = var_to_weights_map['conv1.weight']
    net.params['conv1'][0].data.flat = weights.flat
#     print(net.params['conv1'][0].data.shape)
    # bn_conv1
    # bn_conv1/moving_mean
    mean = var_to_weights_map['bn1.running_mean']
    net.params['conv1/bn'][0].data.flat = mean.flat
    # bn_conv1/moving_variance
    variance = var_to_weights_map['bn1.running_var']
    net.params['conv1/bn'][1].data.flat = variance.flat
    net.params['conv1/bn'][2].data.flat = np.array([1]).flat
    # bn_conv1/gamma
    gamma = var_to_weights_map['bn1.weight']
    net.params['conv1/scale'][0].data.flat = gamma.flat    
    # bn_conv1/beta
    beta = var_to_weights_map['bn1.bias']
    net.params['conv1/scale'][1].data.flat = beta.flat 
    
#     exit()
    # Stage 2  3 4 5 
    for i in range(2,6):
        extract_from_seres10_sebasicblock(net, var_to_weights_map, stage=i)
    
    ##-------C5 _lateral TL6_1 TL6_2 P6----------------
    #c5_lateral.0.weight
    #c5_lateral.0.bias
    #c5_lateral.2.weight
    #c5_lateral.2.bias
    #c5_lateral.4.weight
    #c5_lateral.4.bias
    
    TL6_1_weight = var_to_weights_map['c5_lateral.0.weight']
    TL6_1_bias  = var_to_weights_map['c5_lateral.0.bias']
    
    TL6_2_weight = var_to_weights_map['c5_lateral.2.weight']
    TL6_2_bias  = var_to_weights_map['c5_lateral.2.bias']
    
    P6_weight = var_to_weights_map['c5_lateral.4.weight']
    P6_bias  = var_to_weights_map['c5_lateral.4.bias']
    
    
    net.params['TL6_1'][0].data.flat = TL6_1_weight.flat
    net.params['TL6_1'][1].data.flat = TL6_1_bias.flat
    net.params['TL6_2'][0].data.flat = TL6_2_weight.flat
    net.params['TL6_2'][1].data.flat = TL6_2_bias.flat
    net.params['P6'][0].data.flat = P6_weight.flat
    net.params['P6'][1].data.flat = P6_bias.flat
    
    #C4_lateral  TL5_1 TL5_2
    #c4_lateral.0.weight
    #c4_lateral.0.bias
    #c4_lateral.2.weight
    #c4_lateral.2.bias
    TL5_1_weight = var_to_weights_map['c4_lateral.0.weight']
    TL5_1_bias  = var_to_weights_map['c4_lateral.0.bias']
    
    TL5_2_weight = var_to_weights_map['c4_lateral.2.weight']
    TL5_2_bias  = var_to_weights_map['c4_lateral.2.bias']
    
    net.params['TL5_1'][0].data.flat = TL5_1_weight.flat
    net.params['TL5_1'][1].data.flat = TL5_1_bias.flat
    net.params['TL5_2'][0].data.flat = TL5_2_weight.flat
    net.params['TL5_2'][1].data.flat = TL5_2_bias.flat
    
    #p4_conv P5
    #p4_conv.0.weight
    #p4_conv.0.bias
    p4_conv_weight = var_to_weights_map['p4_conv.0.weight']
    p4_conv_bias = var_to_weights_map['p4_conv.0.bias']
    net.params['P5'][0].data.flat = p4_conv_weight.flat
    net.params['P5'][1].data.flat = p4_conv_bias.flat
    
    ###ARM outout layer
    ### c4_arm_loc_layer->  block_4_1_mbox_loc
    ### c4_arm_cls_layer-> block_4_1_mbox_conf
    ### c5_arm_loc_layer-> block_5_1_mbox_loc 
    ### c5_arm_cls_layer-> block_5_1_mbox_conf
    
    c4_arm_loc_layer_weight = var_to_weights_map['c4_arm_loc_layer.weight']
    c4_arm_loc_layer_bias  = var_to_weights_map['c4_arm_loc_layer.bias']
    net.params['block_4_1_mbox_loc'][0].data.flat = c4_arm_loc_layer_weight.flat
    net.params['block_4_1_mbox_loc'][1].data.flat = c4_arm_loc_layer_bias.flat

    c4_arm_cls_layer_weight = var_to_weights_map['c4_arm_cls_layer.weight']
    c4_arm_cls_layer_bias  = var_to_weights_map['c4_arm_cls_layer.bias']
    net.params['block_4_1_mbox_conf'][0].data.flat = c4_arm_cls_layer_weight.flat
    net.params['block_4_1_mbox_conf'][1].data.flat = c4_arm_cls_layer_bias.flat
    
    c5_arm_loc_layer_weight = var_to_weights_map['c5_arm_loc_layer.weight']
    c5_arm_loc_layer_bias  = var_to_weights_map['c5_arm_loc_layer.bias']
    net.params['block_5_1_mbox_loc'][0].data.flat = c5_arm_loc_layer_weight.flat
    net.params['block_5_1_mbox_loc'][1].data.flat = c5_arm_loc_layer_bias.flat

    c5_arm_cls_layer_weight = var_to_weights_map['c5_arm_cls_layer.weight']
    c5_arm_cls_layer_bias  = var_to_weights_map['c5_arm_cls_layer.bias']
    net.params['block_5_1_mbox_conf'][0].data.flat = c5_arm_cls_layer_weight.flat
    net.params['block_5_1_mbox_conf'][1].data.flat = c5_arm_cls_layer_bias.flat

    ###ODM outout layer
    ### p4_odm_loc_layer-> P5_mbox_loc
    ### p4_odm_cls_layer-> P5_mbox_conf1
    ### p5_odm_loc_layer-> P6_mbox_loc
    ### p5_odm_cls_layer-> P6_mbox_conf1
    
    p4_odm_loc_layer_weight = var_to_weights_map['p4_odm_loc_layer.weight']
    p4_odm_loc_layer_bias  = var_to_weights_map['p4_odm_loc_layer.bias']
    net.params['P5_mbox_loc'][0].data.flat = p4_odm_loc_layer_weight.flat
    net.params['P5_mbox_loc'][1].data.flat = p4_odm_loc_layer_bias.flat
#     print(type(p4_odm_loc_layer_weight),p4_odm_loc_layer_weight.shape,net.params['P5_mbox_loc'][0].data.shape)
    
    p4_odm_cls_layer_weight = var_to_weights_map['p4_odm_cls_layer.weight']
    p4_odm_cls_layer_bias  = var_to_weights_map['p4_odm_cls_layer.bias']
    net.params['P5_mbox_conf1'][0].data.flat = p4_odm_cls_layer_weight.flat
    net.params['P5_mbox_conf1'][1].data.flat = p4_odm_cls_layer_bias.flat
#     print(type(p4_odm_cls_layer_weight),p4_odm_cls_layer_weight.shape,net.params['P5_mbox_conf1'][0].data.shape)
    
#     print(p4_odm_cls_layer_weight)
#     print(net.params['P5_mbox_conf1'][0].data)
    p5_odm_loc_layer_weight = var_to_weights_map['p5_odm_loc_layer.weight']
    p5_odm_loc_layer_bias  = var_to_weights_map['p5_odm_loc_layer.bias']
    net.params['P6_mbox_loc'][0].data.flat = p5_odm_loc_layer_weight.flat
    net.params['P6_mbox_loc'][1].data.flat = p5_odm_loc_layer_bias.flat
#     print(type(p5_odm_loc_layer_weight),p5_odm_loc_layer_weight.shape,net.params['P6_mbox_loc'][0].data.shape)
    
    p5_odm_cls_layer_weight = var_to_weights_map['p5_odm_cls_layer.weight']
    p5_odm_cls_layer_bias  = var_to_weights_map['p5_odm_cls_layer.bias']
    net.params['P6_mbox_conf1'][0].data.flat = p5_odm_cls_layer_weight.flat
    net.params['P6_mbox_conf1'][1].data.flat = p5_odm_cls_layer_bias.flat
#     print(type(p5_odm_cls_layer_weight),p5_odm_cls_layer_weight.shape,net.params['P6_mbox_conf1'][0].data.shape)
    
net = caffe.Net(cf_prototxt, caffe.TRAIN)

torch2caffe(var_to_weights_map, net)
# print(net)
# print(net.params['P5_mbox_conf1'][0].data.shape)
net.save(cf_model)
print('fininsh convert pytorch model to caffemodel'+cf_model)