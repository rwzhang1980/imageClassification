package com.rwzhang.imageClassify;

import java.io.File;
import java.util.Date;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;

import com.rwzhang.imageClassify.constants.Constants;
import com.rwzhang.imageClassify.feature.Hog;
import com.rwzhang.imageClassify.image.ImageHandler;
import com.rwzhang.imageClassify.utils.Utils;

public class Train {

	private Mat trainData = null;
	
	private Mat res_mat = null;
	
	public void train(String posSamplePath, String negSamplePath){
		putSample(posSamplePath, negSamplePath);
		System.out.println("全部样本读取完成!");
		SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);
        svm.setCoef0(0.0);
        svm.setDegree(3);
        svm.setGamma(0);
        svm.setNu(0.5);
        svm.setP(0.1);
        svm.setC(0.01);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, Constants.ITER_MAX_COUNT, Constants.EPSILON));
        Date start = new Date();
        System.out.println("开始训练SVM模型...");
        svm.trainAuto(trainData, Ml.ROW_SAMPLE, res_mat);
        Date end1 = new Date();
        System.out.println("训练完成，耗时：" + (end1.getTime() - start.getTime())/1000 + "秒");
        System.out.println("正在保存训练文件...");
        svm.save(Constants.SVM_FILE_FULL_PATH);
        Date end2 = new Date();
        System.out.println("保存完成，训练结束！耗时：" + (end2.getTime() - end1.getTime())/1000 + "秒");
        
	}
	
	private void putSample(String posSamplePath, String negSamplePath){
		File posDir = new File(posSamplePath);
		File negDir = new File(negSamplePath);
		File[] posFiles = posDir.listFiles();
		File[] negFiles = negDir.listFiles();
		trainData = new Mat(posFiles.length + negFiles.length, Constants.VAR_COUNT, CvType.CV_32FC1);
		res_mat = new Mat(posFiles.length + negFiles.length, 1, CvType.CV_32S);
		putPosSample(posFiles);
		putNegSample(negFiles, posFiles.length);
	}
	
	private void putPosSample(File[] files){
		Date start = new Date();
		System.out.println("正在读取正样本...");
		for(int i = 0;i < files.length;i++){
			Mat mat = Imgcodecs.imread(files[i].getPath());
			float[] data = Hog.openCVGetHog(mat);
			setMatData(data, i);
			res_mat.put(i, 0, Constants.POS_LABEL);
		}
		Date end = new Date();
		System.out.println("正样本读取完成，共读取样本" + files.length + "个，耗时：" + (end.getTime() - start.getTime())/1000 + "秒");
	}
	
	private void putNegSample(File[] files, int startIndex){
		Date start = new Date();
		System.out.println("正在读取负样本...");
		for(int i = 0;i < files.length;i++){
			Mat mat = Imgcodecs.imread(files[i].getPath());
			float[] data = Hog.openCVGetHog(mat);
			setMatData(data, startIndex + i);
			res_mat.put(startIndex + i, 0, Constants.NEG_LABEL);
		}
		Date end = new Date();
		System.out.println("负样本读取完成，共读取样本" + files.length + "个，耗时：" + (end.getTime() - start.getTime())/1000 + "秒");
	}
	
	private void setMatData(float[] data, int index){
		for(int i = 0;i < data.length;i++){
			trainData.put(index, i, data[i]);
		}
	}
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Train train = new Train();
		train.train(Constants.TRAIN_POS_SAMPLE_FOLDER, Constants.TRAIN_NEG_SAMPLE_FOLDER);
		//train.cutNegImage("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\INRIAPerson\\Train\\neg", "D:\\Works\\JavaWorks\\SVM\\ImageClassify\\neg");
	}
}
