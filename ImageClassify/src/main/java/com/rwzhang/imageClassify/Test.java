package com.rwzhang.imageClassify;

import java.io.File;
import java.util.Vector;

import org.bytedeco.javacpp.opencv_stitching.SiftFeaturesFinder;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;

import com.rwzhang.imageClassify.constants.Constants;
import com.rwzhang.imageClassify.utils.Utils;

public class Test {

	public void run(){
		
		
		HOGDescriptor hog = new HOGDescriptor(); // 构建HOG描述子
		Mat svmDetector = buildSVMDetector("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\svm.xml");
		
		//hog.setSVMDetector(svmDetector); // 设置默认SVM分类器
		hog.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector()); // 设置默认SVM分类器
		//hog.load("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\test.xml");
		String path = Constants.TEST_POS_SAMPLE_FOLDER + "\\" + "crop_000004.png"; // 要检测的图片的路径
		File file = new File(path);
		Mat img = Imgcodecs.imread(path); // 读取图片
//		Mat img = Imgcodecs.imread("./data/crowd_2.jpg"); // 读取图片
		if(img.empty()){ // 判断构建的Mat是否为空，如果为空的话是会报错的，为代码强壮性在这里做一个判断
			System.out.println("文件不存在！程序退出！");
			System.exit(0);
		}
		MatOfRect mor = new MatOfRect(); // 检测完毕后会储存在这里
		MatOfDouble mod = new MatOfDouble(); // 不清楚是什么，调用的方法参数里面有，就做个实例吧
		System.out.println("正在检测...");
		hog.detectMultiScale(img, mor, mod, 0, new Size(8, 8), new Size(16, 16), 1.05, 2, false); // 调用方法进行检测
		
		System.out.println("检测完毕！画出矩形...");
		if(mor.toArray().length > 0){ // 判断是否检测到目标对象，如果有就画矩形，没有就执行下一步
			for(Rect r:mor.toArray()){ // 检测到的目标转成数组形式，方便遍历
				r.x += Math.round(r.width*0.1);
				r.width = (int) Math.round(r.width*0.8);
				r.y += Math.round(r.height*0.045);
				r.height = (int) Math.round(r.height*0.85);
				Imgproc.rectangle(img, r.tl(), r.br(), new Scalar(0, 0, 255), 2); // 画出矩形
			}
			System.out.println("矩形绘制完毕！正在输出...");
		}else{
			System.out.println("未检测到目标！绘制矩形失败！输出原文件！");
		}
		
		Imgcodecs.imwrite("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\result\\" + Utils.getFileName(file.getName()) + "." + Utils.getFileNameSuffix(file.getName()), img); // 将已经完成检测的Mat对象写出，参数：输出路径，检测完毕的Mat对象。
		System.out.println("输出完毕！");
	}
	
	private Mat buildSVMDetector(String svmFilePath){
		SVM svm = SVM.load(svmFilePath);
		int varCount = svm.getVarCount();
		int supportVectorCount = svm.getSupportVectors().rows();
		Mat alphaMat = new Mat();
		Mat svidx = new Mat();
		double rho = svm.getDecisionFunction(0, alphaMat, svidx);
		double alphaValue = alphaMat.get(0, 0)[0];
		System.out.println("support_vector_count:" + supportVectorCount);
		System.out.println("var_count:" + varCount);
		System.out.println("rho:" + rho);
		System.out.println("alpha:" + alphaValue);
		
		Mat supportVectorMat = svm.getSupportVectors();
		Mat resultMat = buildResultMat(supportVectorMat, alphaMat);
		resultMat.put(0, 3780, rho);
		return resultMat;
	}
	
	private Mat buildResultMat(Mat supportVectorMat, Mat alphaMat){
		Mat resultMat = new Mat(supportVectorMat.cols() + 1, 1, CvType.CV_32FC1);
		int rows = supportVectorMat.rows();
		int cols = supportVectorMat.cols();
		double alphaValue = alphaMat.get(0, 0)[0];
		for(int i = 0;i < rows;i++){
			for(int j = 0;j < cols;j++){
				double supportVectorValue = supportVectorMat.get(i, j)[0];
				resultMat.put(j, 0, supportVectorValue * alphaValue);
			}
		}
		return resultMat;
	}
	
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Test t = new Test();
		MatOfFloat mf = HOGDescriptor.getDefaultPeopleDetector();
		System.out.println(mf.size());
		System.out.println("rows:" + mf.rows());
		System.out.println("cols:" + mf.cols());
		for(int i = 0;i < mf.rows();i++){
			for(int j = 0;j < mf.cols();j++){
				System.out.println(mf.get(i, j)[0]);
			}
		}
		//t.run();
		//t.buildSVMDetector("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\svm.xml");
	}
}
