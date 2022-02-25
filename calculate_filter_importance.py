#N-mode tensor product를 활용하여 convolution filter의 importance를 quantification
import numpy
import tensorly

def get_prune_list(conv, next_conv, cut_off_rate):
  importance_list = numpy.array([])
  weight = conv.weight.data.cpu().numpy() #pruning할 convolutional layer의 weight tensor 추출
  
  for i in range(conv.out_channels):
    weight_next = next_conv.weight.data.cpu().numpy().squeeze() #후차 convolutional layer의 weight tensor
    target_tensor = tensorly.tenalg.mode_dot(weight.squeeze(),weight_next,0) #N-tensor product를 활용하기 위한 tensor 구하기 (reconstruction error 정의를 위함)
    
    pruned_weight_next = numpy.delete(weight_next, [i], 1) #특정 컨볼루션 필터를 제거
    pruned_weight = numpy.delete(weight.squeeze(), [i], 0) 
    pruned_tensor = tensorly.tenalg.mode_dot(pruned_weight, pruned_weight_next, 0) #reconstruction error를 계산하기 위한 weight tensor를 구성
    
    score = tensorly.norm(target_tensor-pruned_tensor)
    importance_list = numpy.append(importance_list, score * -1.0) 
    prune_list = sorted(range(len(importance_list)),key=lambda i: importance_list[i])[:int(conv.out_channels*cut_off_rate)] #reconstruction error가 작은 순서로 pruning
    return prune_list

