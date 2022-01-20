import numpy as np 
import cv2

cap = cv2.VideoCapture('../data/challenge_video.mp4')


#Camera Matrix
K = np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
 [  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
 [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

#Distortion Coefficients
dist = np.array([[ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05,
		2.20573263e-02]])


def get_yellow_line(img):
	""" Function to extract yellow lane line using HSV values

	Args:
		img : numpy image

	Returns:
		mask : returns binary mask for yellow lane line
	"""
	image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower = np.array([22, 93, 0], dtype="uint8")
	upper = np.array([45, 255, 255], dtype="uint8")

	mask = cv2.inRange(image, lower, upper)
	
	return mask

def region(image):
	"""Defines a region for the detection of lanes

	Args:
		image : numpy image

	Returns:
		mask : returns masked image region
	"""
	height, width = image.shape
	triangle = np.array([
											[(100, height), (475, 325), (width, height)]
											])
	mask = np.zeros_like(image)
	mask = cv2.fillPoly(mask, triangle, 255)
	mask = cv2.bitwise_and(image, mask)
	return mask

def create_coordinates(image, line_parameters):
	"""Generate line coordinates from slope and intercept
	"""
	try:
		slope, intercept = line_parameters
	except :
		return np.array([])
	y1 = image.shape[0]
	y2 = int(y1*0.7)
	x1 = int((y1 - intercept) / slope)
	x2 = int((y2 - intercept) / slope)
	return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
	"""For each set of points fit a line and generate the average of slopes and intercepts
	   for left and right line

	Args:
		image : numpy image
		lines : list of lines detected from hough transform

	Returns:
		[type]: [description]
	"""
	left_fit = []
	right_fit = []
	try:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			
			# It will fit the polynomial and the intercept and slope
			parameters = np.polyfit((x1, x2), (y1, y2), 1)
			slope = parameters[0]
			intercept = parameters[1]
			if slope < 0:
				left_fit.append((slope, intercept))
			else:
				right_fit.append((slope, intercept))
				
		left_fit_average = np.average(left_fit, axis = 0)
		right_fit_average = np.average(right_fit, axis = 0)
		left_line = create_coordinates(image, left_fit_average)
		right_line = create_coordinates(image, right_fit_average)
		return np.array([left_line, right_line])
	except:
		return np.array([])

def draw_on_img(image, lines):
	try:
		x1, y1, x2, y2 = lines[0]
		x3, y3, x4, y4 = lines[1]
		contours = np.array([[x2,y2], [x4,y4], [x3,y3], [x1,y1]])
		poly_image = np.zeros_like(image)
		cv2.fillPoly(poly_image, pts = [contours], color =(255, 0, 0))
		# cv2.line(poly_image, (int(x1), int(y1)), (int(x2),int(y2)), (0,0,255), 3)
		# cv2.line(poly_image, (int(x3), int(y3)), (int(x4),int(y4)), (0,0,255), 3)
		combo_image = cv2.addWeighted(image, 0.98, poly_image, 1, 1)
		return combo_image
	except:
		return image

def get_slope(lines):
	try:
		x1, y1, x2, y2 = lines[0]
		x3, y3, x4, y4 = lines[1]
		tx, ty = (x2 + x4)/2, (y2 + y4)/2
		bx, by = (x3 + x1)/2, (y3 + y1)/2
		slope = (ty - by)/(tx - bx)

		return[tx, ty, bx, by]
	except:
		pass 

	"""Generate output video
	"""
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1200, 617))
if (cap.isOpened()== False): 
	print("Error opening video stream or file")
avg_slope=[]
while(cap.isOpened()):
	
	ret, frame = cap.read()
	
	if ret == True:
		
		img = frame.copy()
		h,  w = img.shape[:2]
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
		# undistort
		dst = cv2.undistort(img, K, dist, None, newcameramtx)
		# crop the image
		x, y, w, h = roi
		dst = dst[y:y+h, x:x+w]
		frame = dst.copy()
		dst[:int(dst.shape[0]*0.66),:]=0
		# get yellow line
		y_mask = get_yellow_line(dst)
		#gray + blur
		img = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
		# img_blur = cv2.GaussianBlur(img, (3,3), 0)

		#threshold
		ret, thresh = cv2.threshold(img,205,255,cv2.THRESH_BINARY)
		
		gray_img = thresh + y_mask
		# optional opening
		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)

		#Edge detection
		edges = cv2.Canny(gray_img, 100, 120)
		# define region  TODO : Possible to replace with Homography

		reg_img = region(edges)

		#Hough transform
		lines = cv2.HoughLinesP(reg_img, rho=6, theta=np.pi/60, threshold=25, minLineLength=40, maxLineGap=150)
		lines = average_slope_intercept(edges, lines)
		centre_line = get_slope(lines)
		
		overlayed_img = draw_on_img(frame, lines)

		try:

			deviation = centre_line[0] - centre_line[2]
			if deviation < 7 and deviation > -7:
				turn_flag = "STRAIGHT"
			if deviation > 7 : 
				turn_flag = "RIGHT"
			if deviation < -7 :
				turn_flag = "LEFT"
			cv2.putText(overlayed_img, turn_flag, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
   
		except:
			pass

		cv2.imshow("processed",overlayed_img)
		out.write(overlayed_img)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	else: 
		break


cap.release()
out.release()
cv2.destroyAllWindows()