package com.rwzhang.imageClassify;

import java.io.File;
import java.util.Date;
import java.util.Vector;

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

public class DetectMultiScale {

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		DetectMultiScale d = new DetectMultiScale();
		d.detectMultiScale(Constants.SVM_FILE_FULL_PATH, Constants.TEST_POS_SAMPLE_FOLDER + "\\" + "crop_000012.png");
		
	}
	
	public void detectMultiScale(String svmfile, String imgpath){
		MatOfFloat detectorMat = buildSVMDetector(svmfile);
		HOGDescriptor hog = new HOGDescriptor();
		hog.setSVMDetector(detectorMat);
		//hog.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());
		//hog.save("d:\\test1.xml");
		File file = new File(imgpath);
		Mat imgMat = Imgcodecs.imread(imgpath);
		MatOfRect mor = new MatOfRect(); // 检测完毕后会储存在这里
		MatOfDouble mod = new MatOfDouble(); // 不清楚是什么，调用的方法参数里面有，就做个实例吧
		Date start = new Date();
		System.out.println("正在检测...");
		//hog.detectMultiScale(imgMat, mor, mod);
		hog.detectMultiScale(imgMat, mor, mod, 0, new Size(8, 8), new Size(16, 16), 1.03, 4, false); // 调用方法进行检测
		if(mor.toArray().length <= 0){
			System.out.println("检测完毕！未发现需要查找对象...");
			return;
		}
		System.out.println("检测完毕！画出矩形...");
		for(Rect r:mor.toArray()){ // 检测到的目标转成数组形式，方便遍历
			r.x += Math.round(r.width * 0.1);
			r.width = (int) Math.round(r.width * 0.8);
			r.y += Math.round(r.height * 0.045);
			r.height = (int) Math.round(r.height * 0.85);
			Imgproc.rectangle(imgMat, r.tl(), r.br(), new Scalar(0, 0, 255), 2); // 画出矩形
		}
		System.out.println("矩形绘制完毕！正在输出...");
		Imgcodecs.imwrite(Constants.OUT_PUT_FOLDER + "\\" + Utils.getFileName(file.getName()) + "." + Utils.getFileNameSuffix(file.getName()), imgMat); // 将已经完成检测的Mat对象写出，参数：输出路径，检测完毕的Mat对象。
		Date end = new Date();
		System.out.println("输出完毕！总耗时" + (end.getTime() - start.getTime())/1000 + "秒");

	}
	
	private MatOfFloat buildSVMDetector(String svmfile){
		SVM svm = SVM.load(svmfile);
		Mat alphaMat = new Mat();
		Mat svidxMat = new Mat();
		double rho = svm.getDecisionFunction(0, alphaMat, svidxMat);
		float[] result = buildResultMat(svm.getSupportVectors(), alphaMat);
		//float[] result = buildResultMat1();
		result[Constants.VAR_COUNT] = (float)rho;
		MatOfFloat f =  new MatOfFloat(result);
		System.out.println(f.get(0, 0)[0]);
		return f;
	}
	
	private float[] buildResultMat1(){
		MatOfFloat mf = HOGDescriptor.getDefaultPeopleDetector();
		int rows = mf.rows();
		int cols = mf.cols();
		float[] result = new float[rows];
		for(int i = 0;i < rows;i++){
			for(int j = 0;j < cols;j++){
				result[i] = (float)mf.get(i, j)[0];
			}
		}
		return result;
	}
	
	private float[] buildResultMat(Mat supportVectorMat, Mat alphaMat){
		float[] result = new float[Constants.VAR_COUNT + 1];
		int rows = supportVectorMat.rows();
		int cols = supportVectorMat.cols();
		for(int i = 0;i < rows;i++){
			float alphaValue = (float)alphaMat.get(i, 0)[0];
			for(int j = 0;j < cols;j++){
				float supportVectorValue = (float)supportVectorMat.get(i, j)[0];
				result[j] = -1 * alphaValue * supportVectorValue;
			}
		}
		return result;
	}
}
