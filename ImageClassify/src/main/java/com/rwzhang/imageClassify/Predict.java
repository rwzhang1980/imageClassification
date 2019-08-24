package com.rwzhang.imageClassify;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.SVM;

import com.rwzhang.imageClassify.constants.Constants;
import com.rwzhang.imageClassify.feature.Hog;

public class Predict {

	private Mat predictData = null;
	
	/**
	 * 开始预测
	 * @param testFilePath
	 */
	public void predict(String predictFilePath){
	    SVM svm = SVM.load(Constants.SVM_FILE_FULL_PATH);
	    predictData = new Mat(1, Constants.VAR_COUNT, CvType.CV_32FC1);
	    putSample(predictFilePath);
		float result = svm.predict(predictData);
		System.out.println(result);
	}
	
	private void putSample(String filePath){
		Mat mat = Imgcodecs.imread(filePath);
		float[] data = Hog.openCVGetHog(mat);
		setMatData(data, 0);
	}
	
	private void setMatData(float[] data, int index){
		for(int i = 0;i < data.length;i++){
			predictData.put(index, i, data[i]);
		}
	}
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Predict p = new Predict();
		p.predict(Constants.TEST_POS_SAMPLE_FOLDER + "\\" + "1.fw.png");
	}
}
