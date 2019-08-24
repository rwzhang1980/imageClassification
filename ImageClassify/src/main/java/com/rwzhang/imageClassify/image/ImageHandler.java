package com.rwzhang.imageClassify.image;

import java.util.ArrayList;
import java.util.List;

import org.bytedeco.javacpp.cvkernels;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * 图像处理
 * @author rwzhang
 *
 */
public class ImageHandler {

	
	/**
	 * 图像转换为灰度
	 * @param sourceMat
	 * @return
	 */
	public static Mat imgRGB2gray(Mat sourceImg){
		Mat retMat = new Mat();
		Imgproc.cvtColor(sourceImg, retMat, Imgproc.COLOR_RGB2GRAY);
		Imgcodecs.imwrite("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\img\\gray.jpg", retMat);
		return retMat;
	}
	
	/**
	 * gamma校正
	 * @param sourceMat
	 * @param gamma
	 * @return
	 */
	public static Mat gammaCorrection(Mat sourceImg, float gamma){
		int width = sourceImg.cols();
		int height = sourceImg.rows();
		byte[] data = new byte[width * height];
		sourceImg.get(0, 0, data);
		int index = 0;
		float i = 0f;
		for(int row = 0;row < height;row++){
			for(int col = 0;col < width;col++){
				index = row * width + col;
				i = data[index] & 0xff;
				i = (i + 0.5f) / 256;
				i = (float)Math.pow(i, gamma);
				i = i * 256 -0.5f;
				data[index] = (byte)i;
			}
		}
		Mat retImg = new Mat();
		sourceImg.copyTo(retImg);
		retImg.put(0, 0, data);
		
		Imgcodecs.imwrite("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\img\\gamma.jpg", retImg);
		return retImg;
	}
	
	/**
	 * 图像裁剪
	 * @param mat
	 * @param pointX
	 * @param pointY
	 * @param width
	 * @param height
	 * @return
	 */
	public static Mat cutImage(Mat sourceImg, int pointX, int pointY, int width, int height){
		int imgWidth = sourceImg.cols();
		int imgHeight = sourceImg.rows();
		if(pointX + width > imgWidth || pointY + height > imgHeight){
			return null;
		}
		Rect rect = new Rect(pointX, pointY, width, height);
		Mat newImg = new Mat(sourceImg, rect);
		Mat retImg = new Mat();
		newImg.copyTo(retImg);
		return retImg;
	}
	
	/**
	 * 图片缩放
	 * @param sourceImg
	 * @param width
	 * @param height
	 * @return
	 */
	public static Mat resize(Mat sourceImg, int width, int height){
		Mat retImg = new Mat();
		Imgproc.resize(sourceImg, retImg, new Size(width, height));
		return retImg;
	}
	
	/**
	 * 翻转图像
	 * @param sourceImg
	 * @param flipCode 0代表垂直方向旋转180度； 1代表水平方向旋转180度；-1代表垂直和水平方向同时旋转
	 * @return
	 */
	public static Mat filp(Mat sourceImg, int flipCode){
		Mat retImg = new Mat();
		Core.flip(sourceImg, retImg, flipCode);
		return retImg;
	}
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat mat = Imgcodecs.imread("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\pos\\crop_000606a.png");
		mat = ImageHandler.imgRGB2gray(mat);
		ImageHandler.gammaCorrection(mat, 1f/2.2f);
	}
	
	
}
