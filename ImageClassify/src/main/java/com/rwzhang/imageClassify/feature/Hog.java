package com.rwzhang.imageClassify.feature;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;

import com.rwzhang.imageClassify.constants.Constants;
import com.rwzhang.imageClassify.entity.PictureGradient;
import com.rwzhang.imageClassify.image.ImageHandler;

public class Hog {

	
	public void start(){
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat sourceMat = Imgcodecs.imread("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\INRIAPerson\\96X160H96\\Train\\pos\\crop_000010b.png");
		sourceMat = preprocessing(sourceMat);
		PictureGradient pictureGradient = gradient(sourceMat);
		float[] ret = acquiredHogDescriptor(sourceMat, pictureGradient, true);
		System.out.println(ret);
	}
	
	/**
	 * 图像预处理
	 * @param sourceMat
	 * @return
	 */
	private Mat preprocessing(Mat sourceMat){
		//图像灰度化
		Mat retMat = ImageHandler.imgRGB2gray(sourceMat);
		//gamma校正
		retMat = ImageHandler.gammaCorrection(retMat, (1f/2.2f));
		return retMat;
	}
	
	private PictureGradient gradient(Mat mat){
		Mat gradientImageX = new Mat();
		Mat gradientImageY = new Mat();
		Imgproc.Sobel(mat, gradientImageX, CvType.CV_64F, 1, 0, 1, 1, 1);
		Imgcodecs.imwrite("D:\\workspace\\SVM\\ImageClassify\\img\\imgX.jpg", gradientImageX);
		Imgproc.Sobel(mat, gradientImageY, CvType.CV_64F, 0, 1, 3, 1, 0);
		Imgcodecs.imwrite("D:\\workspace\\SVM\\ImageClassify\\img\\imgY.jpg", gradientImageY);
		int width = mat.cols();
		int height = mat.rows();
		int index = 0;
		double[] dataGradient = new double[width * height];
        double[] dataGradientX = new double[width * height];
        double[] dataGradientY = new double[width * height];
        gradientImageX.get(0, 0, dataGradientX);
        gradientImageY.get(0, 0, dataGradientY);
		
        PictureGradient pictureGradient = new PictureGradient();
        pictureGradient.setHeight(height);
        pictureGradient.setWidth(width);
        for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				index = row * width + col;
                double gradientX = dataGradientX[index];
                double gradientY = dataGradientY[index];
                dataGradient[index] = Math.sqrt(Math.pow(gradientX, 2) + Math.pow(gradientY, 2));
                double gradientDirection = Math.atan2(gradientY, gradientX) * (180.0 / Math.PI);
 
                pictureGradient.getDirection().add(gradientDirection);
                pictureGradient.getSize().add(dataGradient[index]);
                //System.out.println("index:" + index + ",x方向梯度：" + gradientX + ",y方向梯度：" + gradientY + ",方向梯度：" + dataGradient[index] + ",梯度方向：" + gradientDirection);
			}
        }
        return pictureGradient;
	}
	
	 //特征向量数（bin）
    private static int bin = 9;
    //块包含2个细胞
    private static int blockSection = 2;
    //细胞包含8个像素
    private static int cellSection = 8;
    
	public static float[] acquiredHogDescriptor(Mat imageMat, PictureGradient pictureGradient, boolean overlapFlag) {
        int width = imageMat.width();
        int height = imageMat.height();
 
        //一个block所对应的宽度像素个数
        int blockwidth = blockSection * cellSection;
        //图片的特征
        float[] hogDescriptor = null;
 
        //判断x,y方向可以包含多少个block
        int xBlockSection = width / blockwidth;
        int yBlockSection = height / blockwidth;
//        System.out.println("宽度：" + width + ",xBlockSection:" + xBlockSection);
//        System.out.println("高度：" + height + ",yBlockSection:" + yBlockSection);
 
        //整个图像的梯度方向
        List<Double> direction = pictureGradient.getDirection();
        //整个图像的梯度
        List<Double> size = pictureGradient.getSize();
        int indexBlock, indexCell, indexPixel = 0;
        int count = 0;
        if (overlapFlag) {
            yBlockSection = yBlockSection + yBlockSection * (blockSection - 1) - 1;
            xBlockSection = xBlockSection + xBlockSection * (blockSection - 1) - 1;
            hogDescriptor = new float[xBlockSection * yBlockSection * blockSection * blockSection * bin];
            int currentDescriptorIndex = 0;
            List<double[]> blockFeaturesList = new ArrayList<double[]>();
            //遍历图片（图片中的所有的Block包含覆盖的）
            for (int n = 0; n < yBlockSection; n++) {
                for (int m = 0; m < xBlockSection; m++) {
                    indexBlock = n * xBlockSection * cellSection + m * cellSection;
                    //block中所含细胞的Weight
                    List<double[]> cellWeight = new ArrayList<double[]>();
                    //遍历Block（block中的细胞）
                    for (int i = 0; i < blockSection; i++) {
                        for (int j = 0; j < blockSection; j++) {
                            indexCell = indexBlock + i * blockSection * cellSection + j * cellSection;
                            double[] weight = new double[bin];
                            //遍历细胞（梯度直方图）
                            for (int z = 0; z < cellSection; z++) {
                                for (int x = 0; x < cellSection; x++) {
                                    indexPixel = indexCell + z * cellSection + x;
                                    int directionSection = directionClassification(direction.get(indexPixel));
                                    //根据区间，进行加权,得到细胞的加权数组
                                    weight[directionSection] += size.get(indexPixel);
//                                    System.out.println("index：" + indexPixel + "梯度方向为：" + direction.get(indexPixel) + "梯度大小：" + size.get(indexPixel) + ",梯度区间：" + directionSection);
                                }
                            }
                            //将遍历完的细胞放到cellWeight中，遍历完block中的所有细胞添加到cellWeight中，后进行归一化处理
                            cellWeight.add(weight);
                            count++;
                        }
                    }
                    //将遍历完的block梯度权值进行归一化处理
                    Iterator<double[]> cellWeightIterator = cellWeight.listIterator();
                    float distanceSquare = 0;
                    while (cellWeightIterator.hasNext()) {
                        double[] weight = cellWeightIterator.next();
                        for (double b : weight)
                            distanceSquare += Math.pow(b, 2);
                    }
//                    System.out.println("欧几里得距离的平方：" + distanceSquare);
                    double distance = Math.sqrt(distanceSquare);
//                    System.out.println("欧几里得距离：" + distance);
                    double[] blockFeatures = new double[blockSection * blockSection * bin];
                    int currentIndex = 0;
 
                    cellWeightIterator = cellWeight.listIterator();
                    while (cellWeightIterator.hasNext()) {
                        double[] weight = cellWeightIterator.next();
                        for (double b : weight) {
//                            System.out.println(b + "," + distance);
                            blockFeatures[currentIndex] = b / (distance + 0.01F);
                            currentIndex++;
                        }
                    }
//                    System.out.println(blockFeatures[35]);
                    blockFeaturesList.add(blockFeatures);
                }
            }
 
 
            Iterator<double[]> blockFeaturesListIterator = blockFeaturesList.listIterator();
            while (blockFeaturesListIterator.hasNext()) {
                double[] blockFeatures = blockFeaturesListIterator.next();
                for (double b : blockFeatures) {
                    hogDescriptor[currentDescriptorIndex] = (float) b;
                    currentDescriptorIndex++;
                }
            }
//            for (int i = 0; i < hogDescriptor.length; i++)
//                System.out.println("第" + i + "个，Descriptor：" + hogDescriptor[i]);
 
 
        } else {
            //遍历图片（图片中的Block）
            for (int n = 0; n < yBlockSection; n++) {
                for (int m = 0; m < xBlockSection; m++) {
                    indexBlock = n * xBlockSection * blockwidth + m * blockwidth;
                    //遍历Block（block中的细胞）
                    for (int i = 0; i < blockSection; i++) {
                        for (int j = 0; j < blockSection; j++) {
                            indexCell = indexBlock + i * blockSection * cellSection + j * cellSection;
                            List<int[]> cellWeight = new ArrayList<int[]>();
                            float[] weight = new float[bin];
                            //遍历细胞（梯度直方图）
                            for (int z = 0; z < cellSection; z++) {
                                for (int x = 0; x < cellSection; x++) {
                                    indexPixel = indexCell + z * cellSection + x;
                                    int directionSection = directionClassification(direction.get(indexPixel));
                                    //根据区间，进行加权,得到细胞的加权数组
                                    weight[directionSection] += size.get(indexPixel);
                                    System.out.println("index：" + indexPixel + "梯度方向为：" + direction.get(indexPixel) + ",梯度区间：" + directionSection);
                                }
                            }
                            //将遍历完的细胞放到cellWeight中，遍历完block中的所有细胞添加到cellWeight中，后进行归一化处理
//                            cellWeight.add(weight);
                            count++;
                        }
                    }
                }
            }
        }
//        System.out.println("特征维度:" + count * bin);
        return hogDescriptor;
    }

	/**
     * 限定角度？？
     * 判定梯度方向属于哪一区间
     *
     * @param direction
     */
    public static int directionClassification(double direction) {
        direction = Math.abs(direction);
        //对梯度方向值进行去整，四舍五入
        int dt = new BigDecimal(direction).divide(new BigDecimal(1), 0, BigDecimal.ROUND_HALF_UP).intValue();
        //梯度方向除以单位角度，得到该梯度方向属于哪一区间
        if (dt == 180)
            return bin - 1;
        int unitAngle = 180 / bin;
        int directionSection = dt / unitAngle;
        return directionSection;
    }
	
	public static float[] openCVGetHog(Mat imageMat) {
		//Mat imageMat = Imgcodecs.imread("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\INRIAPerson\\96X160H96\\Train\\pos\\crop_000010b.png");
		//Mat imageMat = Imgcodecs.imread("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\resize\\crop_000010b.png");
		imageMat = ImageHandler.imgRGB2gray(imageMat);
		HOGDescriptor hog = new HOGDescriptor(new Size(Constants.IMG_WIDTH, Constants.IMG_HEIGHT), new Size(16, 16), new Size(8, 8), new Size(8, 8), 9);
        MatOfFloat descriptorsOfMat = new MatOfFloat();
        hog.compute(imageMat, descriptorsOfMat, new Size(8, 8), new Size(0, 0));
        float[] ret = descriptorsOfMat.toArray();
        return ret;
    }
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//Hog hog = new Hog();
		//hog.start();
		Mat mat = Imgcodecs.imread("D:\\Works\\JavaWorks\\SVM\\ImageClassify\\sample\\train\\pos\\crop_000010a.png");
		mat = ImageHandler.imgRGB2gray(mat);
		float[] f = Hog.openCVGetHog(mat);
		System.out.println(f.length);
	}
}
