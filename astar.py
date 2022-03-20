import numpy as np
import cv2
import math

class Node:
	def __init__(self,data,parent,curr_cost,heu_cost):
		self.Node_data = data 			# data = [x,y,angle]
		self.Node_parent = parent
		self.Node_curr_cost = curr_cost
		self.Node_heuristic_cost = heu_cost

def get_line(A,B):
  x1,y1,x2,y2 = A[0],A[1],B[0],B[1]
  m = (y2-y1)/(x2-x1)
  c = y1 - m * x1
  return m,c

def check_for_obstacle_1(points,point): ## returns if the coordinate is within quadrilateral obstacle
  count = 0
  i = 0
  x,y = point[0],point[1]
  m1,c1 = get_line(points[0],points[1])
  m2,c2 = get_line(points[1],points[2])
  m3,c3 = get_line(points[2],points[0])

  a = x*m1 + y*(-1) + c1
  b = x*m2 + y*(-1) + c2
  c = x*m3 + y*(-1) + c3

  check = [a,-b,-c]
  for val in check:
    if (val >= 0):
      count +=1
  if (count == 3):
    i += 1

  count1 = 0
  m11,c11 = get_line(points[0],points[2])
  m22,c22 = get_line(points[2],points[3])
  m33,c33 = get_line(points[3],points[0])

  aa = x*m11 + y*(-1) + c11
  bb = x*m22 + y*(-1) + c22
  cc = x*m33 + y*(-1) + c33

  check1 = [-aa,-bb,cc]
  for vall in check1:
    if(vall <= 0):
      count1 += 1
  if (count1 == 3):
    i += 1
  if (i == 0):
    return None
  else:
    return True

def check_for_obstacle_2(points,point):   ## returns if the coordinate is within hexagonal obstacle
  count = 0
  x,y = point[0],point[1]
  # m1,c1 = get_line(points[0],points[1]) // This doesn't work since division by zero is impossible.
  m2,c2 = get_line(points[1],points[2])
  m3,c3 = get_line(points[2],points[3])
  # m4,c4 = get_line(points[3],points[4]) // This doesn't work since division by zero is impossible.
  m5,c5 = get_line(points[4],points[5])
  m6,c6 = get_line(points[5],points[0])
  
  a = x - 165
  b = x*m2 + y*(-1) + c2
  c = x*m3 + y*(-1) + c3
  d = x - 235
  e = x*m5 + y*(-1) + c5
  f = x*m6 + y*(-1) + c6

  check = [a,b,c,-d,-e,-f]
  for val in check:
    if (val >= 0):
      count += 1
  if (count == 6):
    return True
  else:
    return None

def check_for_obstacle_3(point):    ## returns if the coordinate is within circular obstacle
  x,y = point[0],point[1]
  a = ((x-300)**2) + ((y-185)**2) - (40*40)
  if (a <= 0):
    return True
  else:
    return None

def get_obstacle_coord():

	obs_coord_list=[]
	obstacle_1 = [[36,185],[115,210],[80,180],[105,100]]
	obstacle_2 = [[165,82.5],[165,117.5],[200,139.13],[235,117.5],[235,82.5],[200,60.87]]
	mat_img = np.ones((401,251,3))
	mat = np.ones((406,256))
	for x in range(0,400):
		for y in range (0,251):
			point = [x,y]
			if (check_for_obstacle_1(obstacle_1,point)) or (check_for_obstacle_2(obstacle_2,point)) or (check_for_obstacle_3(point)):
				obs_coord_list.append([x,y])
				mat_img[x][y] = (0,0,0)
				mat[x][y] = 0
	image = cv2.rotate(np.array(mat), cv2.ROTATE_90_COUNTERCLOCKWISE)
	image = cv2.convertScaleAbs(image, alpha=(255.0))
	cv2.imwrite("./docs/obstacle_map.jpg",image)

	return obs_coord_list,mat_img,mat

def euclidean_distance(A,B):
	x1,y1,x2,y2 = A[0],A[1],B[0],B[1]
	dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
	return dist

def get_idx(theta):
	if theta <= 360 and theta >= -360:
		if theta == 0 or theta == 360 or theta == -360:
			idx = 0
			return idx
		if theta == 30 or theta == -330:
			idx = 1
			return idx
		if theta == 60 or theta == -300:
			idx = 2
			return idx
		if theta == 90 or theta == -270:
			idx = 3
			return idx
		if theta == 120 or theta == -240:
			idx = 4
			return idx
		if theta == 150 or theta == -210:
			idx = 5
			return idx
		if theta == 180 or theta == -180:
			idx = 6
			return idx
		if theta == 210 or theta == -150:
			idx = 7
			return idx
		if theta == 240 or theta == -120:
			idx = 8
			return idx
		if theta == 270 or theta == -90:
			idx = 9
			return idx
		if theta == 300 or theta == -60:
			idx = 10
			return idx
		if theta == 330 or theta == -30:
			idx = 11
			return idx
	elif theta > 360:
		while theta >= 360:
			theta = theta - 360
		if theta == 0 or theta == 360:
			idx = 0
			return idx
		if theta == 30 or theta == -330:
			idx = 1
			return idx
		if theta == 60 or theta == -300:
			idx = 2
			return idx
		if theta == 90 or theta == -270:
			idx = 3
			return idx
		if theta == 120 or theta == -240:
			idx = 4
			return idx
		if theta == 150 or theta == -210:
			idx = 5
			return idx
		if theta == 180 or theta == -180:
			idx = 6
			return idx
		if theta == 210 or theta == -150:
			idx = 7
			return idx
		if theta == 240 or theta == -120:
			idx = 8
			return idx
		if theta == 270 or theta == -90:
			idx = 9
			return idx
		if theta == 300 or theta == -60:
			idx = 10
			return idx
		if theta == 330 or theta == -30:
			idx = 11
			return idx
	elif theta < -360:
		while theta <= 360:
			theta = theta + 360
		if theta == 0 or theta == 360:
			idx = 0
			return idx
		if theta == 30 or theta == -330:
			idx = 1
			return idx
		if theta == 60 or theta == -300:
			idx = 2
			return idx
		if theta == 90 or theta == -270:
			idx = 3
			return idx
		if theta == 120 or theta == -240:
			idx = 4
			return idx
		if theta == 150 or theta == -210:
			idx = 5
			return idx
		if theta == 180 or theta == -180:
			idx = 6
			return idx
		if theta == 210 or theta == -150:
			idx = 7
			return idx
		if theta == 240 or theta == -120:
			idx = 8
			return idx
		if theta == 270 or theta == -90:
			idx = 9
			return idx
		if theta == 300 or theta == -60:
			idx = 10
			return idx
		if theta == 330 or theta == -30:
			idx = 11
			return idx

def angle_return(angle):
	if angle >= 360:
		while angle > 360:
			angle = angle - 360
		print (angle)
		return angle
	if angle < -360:
		while angle < -360:
			angle = angle + 360
		return angle
	else:
		return angle

def Action_theta(curr_node,step_size,matrix,goal,theta,matrix_image):
	global count
	global r

	x,y = curr_node.Node_data[0],curr_node.Node_data[1]
	if (x >= (0)) and (x <= (399)) and (y <= (249)) and (y >= (0)):
		x_new = round(step_size*np.cos(np.deg2rad(theta)) + x)
		y_new = round(step_size*np.sin(np.deg2rad(theta)) + y)
		angle = angle_return(theta)
		# print(angle)
		cost = curr_node.Node_curr_cost + step_size
		heu_cost = euclidean_distance([x_new,y_new],goal)
		data = [x_new,y_new,angle]
		
		if x_new < 395 and y_new < 245:
			print(x_new,y_new,angle)
			if matrix[int((x_new - 5)*2)][int((y_new - 5)*2)][:].any() == 0 and matrix[int((x_new)*2)][int((y_new - 5)*2)][:].any() == 0 and matrix[int((x_new + 5)*2)][int((y_new - 5)*2)].any() == 0 and matrix[int((x_new - 5)*2)][int(y_new*2)][:].any() == 0 and matrix[int((x_new + 5)*2)][int(y_new*2)].any() == 0 and matrix[int((x_new + 5)*2)][int((y_new + 5)*2)].any() == 0 and matrix[int((x_new - 5)*2)][int((y_new + 5)*2)].any() == 0 and matrix[int(x_new*2)][int((y_new + 5)*2)].any() == 0:
				new_Node = Node(data,curr_node,cost,heu_cost)
				matrix_image[x_new][y_new] = (0,0,0)
				start_point = (y,x)
				end_point = (y_new,x_new)
				colour = (0,255,0)
				image = cv2.arrowedLine(matrix_image,start_point,end_point,colour,2)
				image = cv2.circle(image, (y,x), r-5, (0,0,255), 1)
				image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
				image = cv2.convertScaleAbs(image, alpha=(255.0))
				cv2.imshow("path",image)
				cv2.imwrite("./docs/images/"+str(count)+".jpg",image)
				image = cv2.circle(image, (y,x), r-5, (255,255,255), 1)
				cv2.waitKey(1)
				count += 1
				return new_Node
			else:
				return False
		else:
			return False
	else:
		print("a")
		return False

def Action(i,curr_node,step_size,matrix,goal,matrix_image):
	if i == 1:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,0,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 2:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,30,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 3:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,60,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 4:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,90,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 5:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,120,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 6:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,150,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 7:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,180,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 8:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,210,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 9:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,240,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 10:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,270,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 11:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,300,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False
	elif i == 12:
		new_Node = Action_theta(curr_node,step_size,matrix,goal,330,matrix_image)
		if new_Node:
			return new_Node
		else:
			return False

def generate_path(node,matrix_image):     ## Bactracking function and generates the path
  
  global count

  path = []
  path = [node.Node_data]
  while node.Node_parent != None:
    node = node.Node_parent
    path.append(node.Node_data)
  path.reverse()
  for i in path:
  	matrix_image[i[0]][i[1]] = (0,0,255)
  	image = cv2.convertScaleAbs(matrix_image.copy(), alpha=(255.0))
  	image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
  	cv2.imshow("final path",image)
  	cv2.imwrite("./docs/images/"+str(count)+".jpg",image)
  	count += 1
  	cv2.waitKey(1)
  	cv2.destroyAllWindows()

  k = path[0]
  x,y = k[0],k[1]
  for i in path:
  	x_new,y_new = i[0],i[1]
  	start_point = (y,x)
  	end_point = (y_new,x_new)
  	colour = (255,0,0)
  	cv2.line(matrix_image,start_point,end_point,colour,1)
  	x = x_new
  	y = y_new
		# image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
		# image = cv2.convertScaleAbs(image, alpha=(255.0))

  image1 = cv2.rotate(matrix_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
  image1 = cv2.convertScaleAbs(image1, alpha=(255.0))
  cv2.imshow("final",image1)
  for i in range(150):
  	cv2.imwrite("./docs/images/"+str(count)+".jpg",image1)
  	count += 1
  cv2.waitKey(1000)
  cv2.destroyAllWindows()

def videowrite():
	global count

	image = cv2.imread("./docs/images/0.jpg")
	h,w,l = image.shape
	size = (w,h)
	video = cv2.VideoWriter("./docs/video.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,size)
	print("generating video...")
	# print(count)
	for i in range(count):
		img = cv2.imread("./docs/images/"+str(i)+".jpg")
		video.write(img)
		# print(i)
	video.release()

def astar(start,goal,obstacles,matrix_image,step,mat):

	queue = []
	cost = []
	state = []
	Visited_region = np.zeros((int(400/0.5),int(250/0.5),12))
	for i in obstacles:
		Visited_region[i[0]*2][i[1]*2][:] = 1

	goal_dist = euclidean_distance([start[0],start[1]],[goal[0],goal[1]])
	startNode = Node(start,None,0,goal_dist)
	# print(startNode.Node_data)
	queue.append(startNode)
	cost.append(startNode.Node_heuristic_cost)
	state.append(startNode.Node_data)

	while queue:
		minimum_cost = min(cost)
		k = cost.index(minimum_cost)
		# print("in queue")

		curr_node = queue.pop(k)
		cost.pop(k)
		state.pop(k)
		visited_idx = get_idx(curr_node.Node_data[2])
		Visited_region[int(curr_node.Node_data[0]*2)][int(curr_node.Node_data[1]*2)][int(visited_idx)] = 1
		if curr_node.Node_data != goal:
			for i in range(1,6):
				new_Node = Action(i,curr_node,step,Visited_region,goal,matrix_image)
				# print("action")
				
				if new_Node != False:
					# print(new_Node.Node_heuristic_cost,new_Node.Node_data[2])
						
					if new_Node.Node_heuristic_cost <= step*1.5 and  new_Node.Node_data[2] == goal[2]:
						print("goal reached")
						return new_Node,matrix_image

					new_Node_idx = int(get_idx(new_Node.Node_data[2]))
					if Visited_region[new_Node.Node_data[0]*2][new_Node.Node_data[1]*2][new_Node_idx] != 1:
						if new_Node.Node_data in state:
							idx = state.index(new_Node.Node_data)
							if new_Node.Node_heuristic_cost < queue[idx].Node_heuristic_cost:
								queue[idx].Node_parent = new_Node.Node_parent
								queue[idx].Node_heuristic_cost = new_Node.Node_heuristic_cost
						else:
							queue.append(new_Node)
							cost.append(new_Node.Node_heuristic_cost)
							state.append(new_Node.Node_data)
							idx = get_idx(new_Node.Node_data[2])
							Visited_region[int(new_Node.Node_data[0]*2)][int(new_Node.Node_data[1]*2)][int(idx)] = 1
							# print(new_Node.Node_heuristic_cost)
	print("Goal Unable to reach at given orientation")
	
if __name__ == '__main__':

	count = 0
	start_x = int(input("Enter Start position X between (0,390): "))
	start_y = int(input("Enter Start position Y between (0,240): "))
	theta_s = int(input("Enter Start orintation theta_s (...,-30,-60,0,30,-60,...): "))
	goal_x = int(input("Enter Goal position X between (0,390): "))
	goal_y = int(input("Enter Goal position Y between (0,240): "))
	theta_g = int(input("Enter Goal orintation theta_g (...,-30,-60,0,30,-60,...): "))
	r = int(input("Enter Robot radius (1,10) : "))
	c = int(input("Enter Obstacle clearance (1,10): "))
	step = int(input("Enter Step size (between 1 mm to 10 mm): "))
	
	obstacles, matrix_image,mat = get_obstacle_coord()

	start = [start_x,start_y,theta_s]
	goal = [goal_x,goal_y,theta_g]

	if mat[start_x][start_y] == 0 or mat[goal_x][goal_y] == 0:
		print("Start position or Goal position in obstacle space")
	else:
		final,image = astar(start,goal,obstacles,matrix_image,step,mat)
		generate_path(final,image)
		videowrite()