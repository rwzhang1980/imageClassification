package com.rwzhang.imageClassify.utils;

import java.io.File;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import com.rwzhang.imageClassify.constants.Constants;
import com.rwzhang.imageClassify.image.ImageHandler;

public class Utils {

	/**
	 * 获取文件名，不包含后缀
	 * @param fileName
	 * @return
	 */
	public static String getFileName(String fileName){
		return fileName.substring(0, fileName.lastIndexOf("."));
	}
	
	/**
	 * 获取文件名后缀
	 * @param fileName
	 * @return
	 */
	public static String getFileNameSuffix(String fileName){
		return fileName.substring(fileName.lastIndexOf(".") + 1);
	}
	
	public static void cutImg(String path, String savepath){
		File dir = new File(path);
		File[] files = dir.listFiles();
		for(File file : files){
			Mat mat = Imgcodecs.imread(file.getPath());
			Mat newMat = ImageHandler.cutImage(mat, 16, 16, 64, 128);
			Imgcodecs.imwrite(savepath + "\\" + file.getName(), newMat);
			System.out.println("正在裁剪：" + file.getPath());
		}
	}
	
	/**
	 * 裁剪负样本图片，一张大负样本图片裁剪成2张
	 * @param sourcePath
	 * @param savePath
	 */
	public static void cutNegImage(String sourcePath, String savePath){
		File sourceDir = new File(sourcePath);
		File[] sourceFiles = sourceDir.listFiles();
		int len = sourceFiles.length;
		for(int i = 0;i < len;i++){
			File sourceFile = sourceFiles[i];
			System.out.println("正在处理[" + (i + 1) + "/" + len + "]" + sourceFile.getPath());
			String fileName = Utils.getFileName(sourceFile.getName());
			String suffix = Utils.getFileNameSuffix(sourceFile.getName());
			Mat negMat = Imgcodecs.imread(sourceFile.getPath());
			int width = negMat.cols();
			int height = negMat.rows();
			int pointY = 0;
			int index = 1;
			while(pointY + Constants.IMG_HEIGHT <= height){
				int pointX = 0;
				while(pointX + Constants.IMG_WIDTH <= width){
					Mat retMat = ImageHandler.cutImage(negMat, pointX, pointY, Constants.IMG_WIDTH, Constants.IMG_HEIGHT);
					if(retMat != null){
						String _savePath = savePath + "\\" + fileName + "_" + (index++) + "." + suffix;
						Imgcodecs.imwrite(_savePath, retMat);
					}
					pointX += Constants.IMG_WIDTH;
				}
				pointY += Constants.IMG_HEIGHT;
			}
		}
	}
	
	public static void flipImg(String sourcePath, String savePath){
		File sourceDir = new File(sourcePath);
		File[] sourceFiles = sourceDir.listFiles();
		int len = sourceFiles.length;
		for(int i = 0;i < len;i++){
			File sourceFile = sourceFiles[i];
			System.out.println("正在处理[" + (i + 1) + "/" + len + "]" + sourceFile.getPath());
			String fileName = Utils.getFileName(sourceFile.getName());
			String suffix = Utils.getFileNameSuffix(sourceFile.getName());
			Mat sourceMat = Imgcodecs.imread(sourceFile.getPath());
			Mat retMat = ImageHandler.filp(sourceMat, 1);
			String _savePath = savePath + "\\" + fileName + "_1." + suffix;
			Imgcodecs.imwrite(_savePath, retMat);
		}
	}
	
	public static void resizePosImage(String sourcePath, String savePath){
		File sourceDir = new File(sourcePath);
		File[] sourceFiles = sourceDir.listFiles();
		int len = sourceFiles.length;
		for(int i = 0;i < len;i++){
			File sourceFile = sourceFiles[i];
			System.out.println("正在处理[" + (i + 1) + "/" + len + "]" + sourceFile.getPath());
			String fileName = Utils.getFileName(sourceFile.getName());
			String suffix = Utils.getFileNameSuffix(sourceFile.getName());
			Mat negMat = Imgcodecs.imread(sourceFile.getPath());
			Mat retMat = ImageHandler.resize(negMat, Constants.IMG_WIDTH, Constants.IMG_HEIGHT);
			String _savePath = savePath + "\\" + fileName + "_1." + suffix;
			Imgcodecs.imwrite(_savePath, retMat);
		}
	}
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Utils.cutNegImage("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\resource\\INRIAPerson\\Train\\neg", "D:\\Works\\JavaWorks\\SVM\\ImageClassify\\sample\\train\\neg");
		//Utils.flipImg("D:/Works/JavaWorks/SVM/ImageClassify/sample/train/neg", "D:/Works/JavaWorks/SVM/ImageClassify/sample/train/neg");
		//Utils.resizePosImage("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\sample\\train\\pos3", "D:\\Works\\JavaWorks\\SVM\\ImageClassify\\sample\\train\\pos4");
		
	}
}
