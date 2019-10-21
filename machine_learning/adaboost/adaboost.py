# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-20 10:04:29
# @Last Modified by:   huzhu
# @Last Modified time: 2019-10-21 09:58:10

from numpy import *
def load_simp_data():
	data_mat = matrix([[1., 2.1],
					   [2., 1.1],
					   [1.3, 1.],
					   [1., 1. ],
					   [2., 1.]])
	class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return data_mat, class_labels

def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
	"""
	@brief      获取判断向量
	@param      data_matrix  输入数据矩阵
	@param      dimen        维度
	@param      thresh_val   阈值
	@param      thresh_ineq  阈值方向，>=或者>
	@return     分类向量
	"""
	ret_array = ones((shape(data_matrix)[0], 1))
	if thresh_ineq == "lt":
		ret_array[data_matrix[:,dimen] <= thresh_val] = -1.0
	else:
		ret_array[data_matrix[:,dimen] > thresh_val] = -1.0
	return ret_array

def build_stump(data_arr, class_labels, D):
	"""
	@brief      创建一个单层决策树
	@param      data_arr      输入数据
	@param      class_labels  数据标签{+1,-1}
	@param      D             权重向量
	@return     单层决策树
	"""
	data_matrix = mat(data_arr)
	label_mat = mat(class_labels).T
	m, n = shape(data_matrix)
	num_step = 10.0  #设定步数
	best_stump = {}  #最佳决策树
	best_class_est = mat(zeros((m, 1)))    #最佳预测标签
	min_error = inf   #最小分类误差，初始化为无穷大
	for i in range(n):
		# 第一层循环，按照维度进行划分
		range_min = data_matrix[:, i].min()
		range_max = data_matrix[:, i].max()
		step_size = (range_max - range_min) / num_step
		for j in range(-1, int(num_step) + 1):
			# 第二层循环，寻找最佳的分类阈值，扩大范围的目的是为了考虑更多的阈值
			for inequal in ["lt", "gt"]:
				thresh_val = (range_min + float(j) * step_size) 
				predict_vals = stump_classify(data_matrix, i, thresh_val, inequal)
				err_arr = mat(ones((m, 1)))
				err_arr[predict_vals == label_mat] = 0
				weight_err = D.T * err_arr  # 分类误差率
				print("split:dim %d, thresh %.2f, thresh inequal :%s, the weighted error is %.3f" \
					  % (i, thresh_val, inequal, weight_err))
				if weight_err < min_error:
					min_error = weight_err
					best_class_est = predict_vals
					best_stump["dimen"] = i 
					best_stump["thresh"] = thresh_val
					best_stump["ineq"] = inequal
	return best_stump, min_error, best_class_est

def adaboost_trains_DS(data_arr, class_labels, numIt = 40):
	"""
	@brief      基于单层决策树的adaboost模型
	@param      data_arr      输入数据
	@param      class_labels  数据标签
	@param      numIt         最大迭代次数
	@return     弱分类器列表
	"""
	weak_class_arr = list()  # 弱分类器列表
	m = shape(data_arr)[0]
	D = mat(ones((m, 1))/ m)  # 初始化权值向量
	agg_class_est = mat(zeros((m, 1)))
	for i in range(numIt):
		# 构建单层决策树
		best_stump, error, class_est = build_stump(data_arr, class_labels, D)
		print("D:", D.T)
		alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
		best_stump["alpha"] = alpha
		weak_class_arr.append(best_stump)
		print("class_est:", class_est.T)
		expon = multiply(-1 * alpha * mat(class_labels).T, class_est)
		D = multiply(D, exp(expon))
		D = D / D.sum()
		# 获取累积分类结果
		agg_class_est += alpha * class_est  
		print("agg_class_est: ", agg_class_est.T)
		# 计算累积错误率
		agg_errors = multiply(sign(agg_class_est) != mat(class_labels).T, ones((m, 1)))
		error_rate = agg_errors.sum() / m
		print("total error: ", error_rate)
		if error_rate == 0.0:
			break
	return weak_class_arr

def ada_classify(data_to_class, classifier_arr):
	"""
	@brief      分类测试函数
	@param      data_to_class   The data to class
	@param      classifier_arr  The classifier arr
	@return     { description_of_the_return_value }
	"""
	data_matrix = mat(data_to_class)
	m = shape(data_matrix)[0]
	agg_class_est = mat(zeros((m, 1)))
	for i in range(len(classifier_arr)):
		class_est = stump_classify(data_matrix, classifier_arr[i]["dimen"], \
								   classifier_arr[i]["thresh"], \
								   classifier_arr[i]["ineq"])
		agg_class_est += classifier_arr[i]["alpha"] * class_est
		print(agg_class_est)
	return sign(agg_class_est)

def plot_roc(pred_strengths, class_labels):
	"""
	@brief      绘制ROC曲线
	@param      pred_strengths  The predicate strengths
	@param      class_labels    The class labels
	@return     { description_of_the_return_value }
	"""
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep


    
if __name__ == '__main__':
	data_mat, class_labels = load_simp_data()
	D = mat(ones((5, 1)) / 5)
	#build_stump(data_mat, class_labels, D)
	classifier_array = adaboost_trains_DS(data_mat, class_labels,9)
	predict = ada_classify([0, 0], classifier_array)
	print(predict)

