import tensorflow as tf

hello = tf.constant('Hello tensorfolw')
sess = tf.Session()
print(sess.run(hello))