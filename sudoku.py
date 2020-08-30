from cv2 import cv2
import numpy as np
import random

image_to_show = [[]]
undo_stack = []
redo_stack = []
markings = [0] * 81


def display_image(image_title, img):
#imS = cv2.resize(img, (1100, 650))  
    cv2.imshow(image_title, img)
    return


def preprocess_image(img):
    # Gaussian blur using a 9*9
    blur = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Dilating the image
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    th = cv2.dilate(th, kernel)
    return th


def find_corners_of_largest_polygon(original, img):

    imgCopy = original.copy()
    cnts, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    '''c = cnts[2]
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)'''
    min_area = np.inf

    for c in cnts:
        # approximate the contour

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # screenCnt = [[]]
        if len(approx) == 4:

            area = cv2.contourArea(c)
            if area < min_area:
                # del screenCnt[::]
                # screenCnt = approx

                (x, y, w, h) = cv2.boundingRect(approx)

                if ((int(w/h)) == 1):

                    screenCnt = approx
                    min_area = area

    cv2.drawContours(imgCopy, [screenCnt], -1, (0, 255, 0), 3)

    # Top-left corner has minimum x+y
    s = []
    for x in screenCnt:
        for y in x:
            s.append(sum(y))

    d = np.array([])
    for x in screenCnt:
        for y in x:
            d = np.append(d, np.diff(y))

    rect = np.zeros((4, 2), dtype="float32")
    # Minimum sum is top-left corner
    rect[0] = screenCnt[s.index(min(s))]
    # Maximum is bottom-right
    rect[2] = screenCnt[s.index(max(s))]
    # Minimum difference is top-right point
    rect[1] = screenCnt[np.argmin(d)]
    # maximum difference is bottom left
    rect[3] = screenCnt[np.argmax(d)]

    # Thus the order of the points is top-left, top-right, bottom-right,bottom-left
    # print("rectangle", rect)
    # cv2.drawContours(imgCopy, [c], -1, (0, 255, 0), 3)
    # display_image('conotur waala image', imgCopy)
    return rect


def find_side_of_square(pts):
    side1 = np.sqrt((pts[1][0] - pts[0][0])**2 + (pts[1][1] - pts[0][1])**2)
    side2 = np.sqrt((pts[2][0] - pts[1][0])**2 + (pts[2][1] - pts[1][1])**2)
    side3 = np.sqrt((pts[3][0] - pts[2][0])**2 + (pts[3][1] - pts[2][1])**2)
    side4 = np.sqrt((pts[3][0] - pts[0][0])**2 + (pts[3][1] - pts[0][1])**2)

    # length = max(side1, side3)
    # breadth = max(side2, side4)
    side = max(side1, side2, side3, side4)

    return side


def infer_grid(side, corners, img):
    squares = []
    side = side / 9

    for x in corners[0:1]:
        x_shift = x[0]
        y_shift = x[1]

    for j in range(9):
        for i in range(9):
            # Top left corner of a bounding box
            p1 = ((i * side) + x_shift, (j * side) + y_shift)
            # Bottom right corner of bounding box
            p2 = (((i + 1) * side) + x_shift, ((j + 1) * side) + y_shift)
            squares.append((p1, p2))
    return squares


def draw_rectangle(squares, img):
    i = 0
    for x in squares:

        for y in x:
            if i % 2 == 0:
                top_left = (int(y[0]), int(y[1]))
            else:
                bottom_right = (int(y[0]), int(y[1]))
            i += 1

        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    # display_image(' image with rectangles', img)
    return


def redo_operation(squares, img, original_image):

    if len(redo_stack) == 0:
        return
    i, number = redo_stack.pop()
    display_numbers_on_grid(i, number, squares, img, original_image)
    return


def undo_operation(squares, img, original_image):

    if len(undo_stack) == 0:
        return

    i, number, category = undo_stack.pop()
    if category == 'normal' or category == 'pencil_mark':
        redo_stack.append((i, number))
  

    side_of_square = int(squares[i+1][0] - squares[i][0])
    top_left = squares[i]
    x, y = top_left
    x, y = int(x), int(y)
    tup = (x+10, y+40)

    if category == 'normal':
        for p in range(x, x+side_of_square):
            for q in range(y, y+side_of_square):
                img[q, p] = original_image[q, p]

        markings[int(i/2)] = 0
        temp_tup = ()
        flag = False
        stack_copy = undo_stack[::-1]
        for element in stack_copy:
            if element[2] == 'pencil_mark':
                display_pencil_markings(side_of_square, element[1], img, x, y)
                temp_tup = temp_tup + (element[1],)
                flag = True
            else:
                break

        if flag == True:
            markings[int(i/2)] = temp_tup[::-1]

        '''if ((i, number, category) in undo_stack):
            print("james", (i, number, category))
            print("james2", tup)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(number), tup,
                        font, 1, (255, 0, 0), 2, cv2.LINE_AA)'''

    elif category == 'pencil_mark':
        if number == 1 or number == 4 or number == 7:
            x = x + int(side_of_square/9)
        elif number == 2 or number == 5 or number == 8:
            x = x + int(side_of_square/2.6)
        elif number == 3 or number == 6 or number == 9:
            x = x + int(side_of_square/1.5)

        if number == 1 or number == 2 or number == 3:
            y = y + int(side_of_square/9.4)
        elif number == 4 or number == 5 or number == 6:
            y = y + int(side_of_square/2.7)
        elif number == 7 or number == 8 or number == 9:
            y = y + int(side_of_square/1.5)

        for p in range(x, x+15):
            for q in range(y, y+15):
                img[q, p] = original_image[q, p]

        stack_copy = undo_stack[::-1]
        if stack_copy[0][2] == 'normal':
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(stack_copy[0][1]), tup,
                        font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        copy = markings[int(i/2)]
        markings[int(i/2)] = markings[int(i/2)][:-1]
        if len(markings[int(i/2)]) == 0:
            markings[int(i/2)] = copy[0]

    elif category == 'delete_normal':
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(number), tup,
                    font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        markings[int(i/2)] = number

    elif category == 'delete_pencil_mark':
        display_pencil_markings(side_of_square, number, img, x, y)
        if markings[int(i/2)] == 0:
            markings[int(i/2)] = (number,)
        else:
            markings[int(i/2)] = markings[int(i/2)] + (number,)


    return


def delete_operation(i, squares, img, original_image):

    global undo_stack
    side_of_square = int(squares[i+1][0] - squares[i][0])
    top_left = squares[i]
    x, y = top_left
    x, y = int(x), int(y)

    for p in range(x, x+side_of_square):
        for q in range(y, y+side_of_square):
            img[q, p] = original_image[q, p]

    if isinstance(markings[int(i/2)], tuple):
        for el in markings[int(i/2)]:
            maintain_stack(i, el, 'delete_pencil_mark')
    elif markings[int(i/2)] >= 1 or markings[int(i/2)] <= 9:
        maintain_stack(i, markings[int(i/2)], 'delete_normal')

    '''print("before delete undo", undo_stack)
    if markings[int(i/2)] >= 1 or markings[int(i/2)] <= 9:
        print("testing 123")
        maintain_stack(i, markings[int(i/2)], 'normal')
    elif isinstance(markings[int(i/2)], tuple):
        for el in markings[int(i/2)]:
            maintain_stack(i, el, 'pencil_mark')'''

    markings[int(i/2)] = 0

    # copy_stack = [[k for k in undo_stack if k[0] == i]]
    '''for el in copy_stack:
        redo_stack.append((el[0], el[1]))'''

    # undo_stack = [k for k in undo_stack if k[0] != i]

    '''temp = [0] * len(undo_stack)
    k = 0
    print(i)
    for el in undo_stack:
        print(el)
        if(el[0] == i):
            temp[k] = 1
        k += 1
    print(temp)

    for j in range(len(temp)):
        if temp[j] == 1:
            print("value of j", j)
            print(undo_stack.index(undo_stack[j]))'''
    '''r, number, category = undo_stack.pop(
            undo_stack.index(undo_stack[j]))

        redo_stack.append((r, number))'''

    # maintain_stack(i, )

    #print("after delete redo", redo_stack)

    return


def display_pencil_markings(side_of_square, number, img, x, y):

   
    font = cv2.FONT_HERSHEY_PLAIN
    if number == 1:
        cv2.putText(img, str(number), (x+int(side_of_square/9), y+int(side_of_square/2.8)),
                    font, 1, (0, 0, 255), 2, cv2.LINE_4)
    elif number == 2:
        cv2.putText(img, str(number), (x+int(side_of_square/2.6), y+int(side_of_square/2.8)),
                    font, 1, (0, 0, 255), 2, cv2.LINE_4)
    elif number == 3:
      
        cv2.putText(img, str(number), (x+int(side_of_square/1.5), y+int(side_of_square/2.8)),
                    font, 1, (0, 0, 255), 2, cv2.LINE_4)
    elif number == 4:
        cv2.putText(img, str(number), (x+int(side_of_square/9), y+int(side_of_square/1.6)),
                    font, 1, (0, 0, 255), 2, cv2.LINE_4)
    elif number == 5:
        cv2.putText(img, str(number), (x+int(side_of_square/2.6), y+int(side_of_square/1.6)),
                    font, 1, (0, 0, 255), 2, cv2.LINE_4)
    elif number == 6:
        cv2.putText(img, str(number), (x+int(side_of_square/1.5), y+int(side_of_square/1.6)),
                    font, 1, (0, 0, 255), 2, cv2.LINE_4)
    elif number == 7:
        cv2.putText(img, str(number), (x+int(side_of_square/9), y+int(side_of_square/1.1)),
                    font, 1, (0, 0, 255), 2, cv2.LINE_4)
    elif number == 8:
        cv2.putText(img, str(number), (x+int(side_of_square/2.6), y+int(side_of_square/1.1)),
                    font, 1, (0, 0, 255), 2, cv2.LINE_4)
    elif number == 9:
        cv2.putText(img, str(number), (x+int(side_of_square/1.5), y+int(side_of_square/1.1)),
                    font, 1, (0, 0, 255), 2, cv2.LINE_4)

    return


def maintain_stack(i, number, category):
    undo_stack.append((i, number, category))
    return


def display_numbers_on_grid(i, key, squares, img, original_image):

    sudoku_matrix = []

    for j in range(81):
        sudoku_matrix.append(0)

    if key == ord('1'):
        number = 1
    elif key == ord('2'):
        number = 2
    elif key == ord('3'):
        number = 3
    elif key == ord('4'):
        number = 4
    elif key == ord('5'):
        number = 5
    elif key == ord('6'):
        number = 6
    elif key == ord('7'):
        number = 7
    elif key == ord('8'):
        number = 8
    elif key == ord('9'):
        number = 9
    else:
        number = key

    tup = squares[i]
    x, y = tup
    x, y = int(x), int(y)
    tup = (x+10, y+40)

    # print("side", squares[i+1][0] - squares[i][0])
    side_of_square = int(squares[i+1][0] - squares[i][0])

    # If the cell is previously blank or the number keyed is a previously existing pencilmark on the cell
    if (markings[int(i/2)] == 0 or (isinstance(markings[int(i/2)], tuple) and number in markings[int(i/2)])):

        # Cell is blank
        if(markings[int(i/2)] == 0):
            markings[int(i/2)] = number
            maintain_stack(i, number, 'normal')

        # Number previously present as pencil mark
        if (isinstance(markings[int(i/2)], tuple) and number in markings[int(i/2)]):
            markings[int(i/2)] = number
            for p in range(x, x+side_of_square):
                for q in range(y, y+side_of_square):
                    img[q, p] = original_image[q, p]
            maintain_stack(i, number, 'normal')

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(number), tup,
                    font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    elif (isinstance(markings[int(i/2)], tuple) or (markings[int(i/2)] <= 9 and markings[int(i/2)] >= 1)):

        if markings[int(i/2)] == number:
            markings[int(i/2)] = (number,)

            for p in range(x, x+side_of_square):
                for q in range(y, y+side_of_square):
                    img[q, p] = original_image[q, p]
            maintain_stack(i, number, 'pencil_mark')

        elif ((isinstance(markings[int(i/2)], tuple) == False) and (markings[int(i/2)] <= 9 and markings[int(i/2)] >= 1)):

            for p in range(x, x+side_of_square):
                for q in range(y, y+side_of_square):
                    img[q, p] = original_image[q, p]
            display_pencil_markings(
                side_of_square, markings[int(i/2)], img, x, y)

            maintain_stack(i, markings[int(i/2)], 'pencil_mark')
            maintain_stack(i, number, 'pencil_mark')

            markings[int(i/2)] = (markings[int(i/2)], number)

        elif (isinstance(markings[int(i/2)], tuple)):
            markings[int(i/2)] = markings[int(i/2)] + (number,)
            maintain_stack(i, number, 'pencil_mark')

        display_pencil_markings(side_of_square, number, img, x, y)

  
    # print("mark", markings)

    # display_image("kjdk", img)
    return


def draw_rectangle_on_key_press(squares, img):
    """This functions handles the drawing on rectangles on screen on keypress.
    It does so by taking the top left and bottom right points of the rectangle and drawing it onto image"""
    global image_to_show
    squares = [item for t in squares for item in t]
    image_to_show = np.copy(img)
    original_image = np.copy(img)

    finish = False
    i = -2
    # skip_key = False
    while not finish:
        display_image('suDOku', image_to_show)

        # if not skip_key:
        key = cv2.waitKey(0)

        # right key
        if key == ord('d'):
            i += 2
            if i % 18 == 0:
                i -= 18

        elif key == ord('a'):
            i -= 2
            if i % 18 == 16:
                i += 18

        elif key == ord('s'):
            i += 18
            if i >= 162:
                i -= 162

        elif key == ord('w'):
            i -= 18
            if i < 0:
                i += 162

        elif key == ord('1') or key == ord('2') or key == ord('3') or key == ord('4') or key == ord('5')or key == ord('6')or key == ord('7')or key == ord('8')or key == ord('9'):
            # cv2.setMouseCallback("moving rectangles",
            # detect_mouse_coordinates, (squares, img))
            display_numbers_on_grid(i, key, squares, img, original_image)

        elif key == ord('u'):
            undo_operation(squares, img, original_image)

        elif key == ord('r'):
            redo_operation(squares, img, original_image)

        elif key == ord('m'):
            delete_operation(i, squares, img, original_image)

        elif key == 27:
            finish = True
        '''if mouse_coordinates[0] != -1 and mouse_coordinates[1] != -1:
            skip_key = True
        else:
            skip_key = False'''

        # mouse_coordinates = (-1, -1)
        # erasing previously drawn triangles by creating clone of image
        image_to_show = np.copy(img)
        top_left = (int((squares[i])[0]), int((squares[i])[1]))
        bottom_right = (int((squares[i+1])[0]), int((squares[i+1])[1]))

        def detect_mouse_coordinates(event, x, y, flags, param):
            # global image_to_show
            nonlocal i
            mouse_coordinates = (0, 0)
            squares = param[0]
            img = param[1]

            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_coordinates = (x, y)
                # print("mouse", mouse_coordinates)
                for j in range(161):
                    point = squares[j]
                    next_point = squares[j+1]

                    if ((mouse_coordinates[0] > point[0] and mouse_coordinates[0] < next_point[0]) and (mouse_coordinates[1] > point[1] and mouse_coordinates[1] < next_point[1])):
                        point = (int(point[0]), int(point[1]))
                        next_point = (int(next_point[0]), int(next_point[1]))
                        break

                for k in range(161):
                    squares[k] = (int(squares[k][0]), int(squares[k][1]))
                    squares[k+1] = (int(squares[k+1][0]), int(squares[k+1][1]))

                    if(squares[k] == point and squares[k+1] == next_point):
                        break

                i = k
                top_left = point
                bottom_right = next_point
                image_to_show = np.copy(img)
                cv2.rectangle(image_to_show, top_left,
                              bottom_right, (0, 255, 0), 3)
                display_image('moving rectangles', image_to_show)

        cv2.setMouseCallback(
            "moving rectangles", detect_mouse_coordinates, (squares, img))

        cv2.rectangle(image_to_show, top_left,
                      bottom_right, (0, 255, 0), 3)
    return


def main(image_path):
    # Reading image
    original_rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # display_image('original image', original)

    processed = preprocess_image(original)

    corners = find_corners_of_largest_polygon(original_rgb, processed)

    side = find_side_of_square(corners)
    squares = infer_grid(side, corners, original)
    draw_rectangle_on_key_press(squares, original_rgb)
    # print(squares)
    # print(corners, side)
    # display_image('processed image', processed)

    return


if __name__ == '__main__':
    image_path = 'images/classic.png'
    main(image_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
