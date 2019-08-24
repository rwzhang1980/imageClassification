package com.rwzhang.imageClassify.entity;

import java.util.ArrayList;
import java.util.List;

public class PictureGradient {

	private int height;
	
	private int width;
	
	private List<Double> direction = new ArrayList<Double>();
	
	private List<Double> size = new ArrayList<Double>();

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	public List<Double> getDirection() {
		return direction;
	}

	public List<Double> getSize() {
		return size;
	}
}
