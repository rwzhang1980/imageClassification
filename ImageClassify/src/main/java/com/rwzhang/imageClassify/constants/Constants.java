package com.rwzhang.imageClassify.constants;

public class Constants {

	public static int IMG_WIDTH = 64;
	
	public static int IMG_HEIGHT = 128;
	
	public static String BASE_PATH = "D:/Works/JavaWorks/SVM/ImageClassify"; //基础目录
	
	public static String SAMPLE_FOLDER = BASE_PATH + "/sample";//样本目录
	
	public static String TRAIN_SAMPLE_FOLDER = SAMPLE_FOLDER + "/train";//训练样本目录
	
	public static String TRAIN_POS_SAMPLE_FOLDER = TRAIN_SAMPLE_FOLDER + "/pos";//训练正向样本目录
	
	public static String TRAIN_NEG_SAMPLE_FOLDER = TRAIN_SAMPLE_FOLDER + "/neg";//训练反向样本目录
	
	public static String TEST_SAMPLE_FOLDER = SAMPLE_FOLDER + "/test";//测试样本目录
	
	public static String TEST_POS_SAMPLE_FOLDER = TEST_SAMPLE_FOLDER + "/pos";//测试正向样本目录
	
	public static String TEST_NEG_SAMPLE_FOLDER = TEST_SAMPLE_FOLDER + "/neg";//测试反向样本目录
	
	public static String SVM_FILE = "svm.xml";
	
	public static String SVM_FILE_FULL_PATH = BASE_PATH + "/" + SVM_FILE;
	
	public static int ITER_MAX_COUNT = 1000;
	
	public static double EPSILON = 1e-3;//1e-6
	
	public static int VAR_COUNT = 3780;
	
	public static int POS_LABEL = 1;
	
	public static int NEG_LABEL = -1;
	
	public static String OUT_PUT_FOLDER = BASE_PATH + "/result";
	
	static{
		if(IMG_WIDTH == 64 && IMG_HEIGHT == 128){
			VAR_COUNT = 3780;
		}else if(IMG_WIDTH == 48 && IMG_HEIGHT == 96){
			VAR_COUNT = 1980;
		}
	}
}
