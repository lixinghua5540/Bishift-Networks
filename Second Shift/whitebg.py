import cv2


epoch = 600
miss = 'results/exp/test_' + str(epoch) + '/images/' + '5_Missing.png'
mask = 'masks/testing_masks/mask1.png'
miss = cv2.imread(miss)
mask = cv2.imread(mask)
img = cv2.add(miss, mask)
cv2.imwrite('results/exp/test_' + str(epoch) + '/images/' + '5_Missing.png', img)