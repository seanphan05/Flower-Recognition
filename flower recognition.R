# import dataset and create directories
dir_main       <- "~/Desktop/Deep Learning with R/dataset/flowers"
dir_train      <- file.path(dir_main, "train")
dir_validation <- file.path(dir_main, "validation")
dir_test       <- file.path(dir_main, "test")

dir_train_daisy     <- "~/Desktop/Deep Learning with R/dataset/flowers/train/daisy"
dir_train_dandelion <- "~/Desktop/Deep Learning with R/dataset/flowers/train/dandelion"
dir_train_rose      <- "~/Desktop/Deep Learning with R/dataset/flowers/train/rose"
dir_train_sunflower <- "~/Desktop/Deep Learning with R/dataset/flowers/train/sunflower"
dir_train_tulip     <- "~/Desktop/Deep Learning with R/dataset/flowers/train/tulip"

dir_validation_daisy     <- "~/Desktop/Deep Learning with R/dataset/flowers/validation/daisy"
dir_validation_dandelion <- "~/Desktop/Deep Learning with R/dataset/flowers/validation/dandelion"
dir_validation_rose      <- "~/Desktop/Deep Learning with R/dataset/flowers/validation/rose"
dir_validation_sunflower <- "~/Desktop/Deep Learning with R/dataset/flowers/validation/sunflower"
dir_validation_tulip     <- "~/Desktop/Deep Learning with R/dataset/flowers/validation/tulip"

dir_test_daisy     <- "~/Desktop/Deep Learning with R/dataset/flowers/test/daisy"
dir_test_dandelion <- "~/Desktop/Deep Learning with R/dataset/flowers/test/dandelion"
dir_test_rose      <- "~/Desktop/Deep Learning with R/dataset/flowers/test/rose"
dir_test_sunflower <- "~/Desktop/Deep Learning with R/dataset/flowers/test/sunflower"
dir_test_tulip     <- "~/Desktop/Deep Learning with R/dataset/flowers/test/tulip"

###################################################################
########### set up model from scratch with augmentation ###########

library(keras)
# Defining a convnet that includes dropout:
s.model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")

s.model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("acc"))

# add augmentation factor to train datagen:
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,zoom_range = 0.2,
  horizontal_flip = TRUE )
# Note that the validation data shouldn't be augmented!
test_datagen <- image_data_generator(rescale = 1/255)

# read images from directories
train_generator <- flow_images_from_directory(
  dir_train,
  datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical",
  shuffle=TRUE )

validation_generator <- flow_images_from_directory(
  dir_validation,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical",
  shuffle=FALSE )

# fitting model using a batch generator
hist <- s.model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50 )
hist


# create test_generator:
test_generator <- flow_images_from_directory(
  dir_test,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)

# return class labels in train_generator:
test_generator$classes
train_generator$
  
# return the accuracy and loss of test generator
eval_s.model <- s.model %>% evaluate_generator(test_generator, step=50)
eval_s.model
# return the probability of prediction
pred_s.model <- s.model %>% predict_generator(test_generator, step=50)
pred_class_s.model <- max.col(pred_s.model)-1


pred_class_s.model
s.model_accuracy = length(which(pred_class_s.model==test_generator$classes))/length(test_generator$classes)
s.model_accuracy

s.model %>% save_model_hdf5("flower_recognition.h5")

########### set up model from scratch with augmentation ###########
###################################################################

###################################################################
################ using feature extraction in model ################

library(keras)
# Instantiating the VGG16 convolutional base from keras
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# Extract features using the pretrained convolutional base
datagen <- image_data_generator(rescale = 1/255)
batch_size <- 20
num_class = 5
extract_features <- function(directory, sample_count) {
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count, num_class))
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "categorical")
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range,] <- labels_batch
    i <- i + 1
    # we set up the break so that each image has been read one
    if (i * batch_size >= sample_count)
      break
  }
  list(
    features = features,
    labels = labels
  )
}

# extract train, validation, and test data features
train <- extract_features(dir_train, 5500)
validation <- extract_features(dir_validation, 1000)
test <- extract_features(dir_test, 1000)

# flatten extracted features:
# function to reshape features to fit input densely connected classifier
reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}

# reshape train, validation, and test features
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

# Defining and training the densely connected classifier
model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = 4 * 4 * 512) %>%
  layer_dropout(rate = 0.5) %>%                  # add dropout point
  layer_dense(units = 5, activation = "softmax") # softmax is used for multi class

# Configuring the model for training
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "categorical_crossentropy",             # loss function for multi class
  metrics = c("accuracy")
)

# Fitting the model with early setting
history <- model %>% fit(
  train$features, train$labels,
  epochs = 30,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels)
)
history

# Implement model on test data and evaluate model
# return the accuracy and loss 
eval_results <- model %>% evaluate(test$features, test$labels)
# return the prediction for classes of test data
pred_class   <- model %>% predict_classes(test$features) %>% as.vector()
# Exam the accuracy
test$classes <- max.col(test$labels)-1 # reverse one-hot classes
length(which(pred_class == test$classes))

# Create confusion matrix
library(gmodels)
table(pred_class, test$classes)
CrossTable(test$classes, pred_class,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual', 'predicted'))

################ using feature extraction in model ################
###################################################################

###################################################################
######################## Visulaization ############################

##### Visualizing intermediate activations

# load the model
vis.model <- load_model_hdf5("flower_recognition.h5")

# Preprocessing a single image
img_path <- "~/Desktop/Deep Learning with R/dataset/flowers/test/daisy/510844526_858b8fe4db.jpg"
img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255
dim(img_tensor)

# display a picture 
plot(as.raster(img_tensor[1,,,]))

# Extracts the outputs of the top eight layers
layer_outputs <- lapply(vis.model$layers[1:8], function(layer) layer$output)
# Creates a model that will return these outputs, given the model input
activation_model <- keras_model(inputs = vis.model$input, outputs = layer_outputs)

# Running the model in predict mode
activations <- activation_model %>% predict(img_tensor)

first_layer_activation <- activations[[1]]
dim(first_layer_activation)

# plot_channel function
plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1,
        col = terrain.colors(12))
}
plot_channel(first_layer_activation[1,,,7])

# Visualizing every channel in every intermediate activation
image_size <- 58
images_per_row <- 16
for (i in 1:8) {
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  png(paste0("daisy_activations_", i, "_", layer_name, ".png"),
      width = image_size * images_per_row,
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  par(op)
  dev.off()
}

##### Visualizing heatmaps of class activation

# loading the VGG16 network with pretrained weights and input shape
vis.model1 <- application_vgg16(weights = "imagenet")

# Preprocessing an input image for VGG16
img_path <- "~/Desktop/Deep Learning with R/dataset/flowers/test/daisy/1031799732_e7f4008c03.jpg"
img <- image_load(img_path, target_size = c(224, 224)) %>%
  image_to_array() %>%
  array_reshape(dim = c(1, 224, 224, 3)) %>%
  imagenet_preprocess_input()

# list top three classes predicted for the image
preds <- vis.model1 %>% predict(img)
imagenet_decode_predictions(preds, top = 3)[[1]]

# return the daisy class index
daisy.index <- which.max(preds[1,])

# set up the Grad-CAM process:
daisy_output <- vis.model1$output[, daisy.index]
last_conv_layer <- vis.model1 %>% get_layer("block5_conv3")
grads <- k_gradients(daisy_output, last_conv_layer$output)[[1]]
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
iterate <- k_function(list(vis.model1$input),
                      list(pooled_grads, last_conv_layer$output[1,,,]))
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))
for (i in 1:512) {
  conv_layer_output_value[,,i] <-
  conv_layer_output_value[,,i] * pooled_grads_value[[i]]
}
heatmap <- apply(conv_layer_output_value, c(1,2), mean)

heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)
write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}
write_heatmap(heatmap, "daisy_heatmap.png")

# create the daisy heatmap to visualize which parts of image are the most daisy-like
library(magick)
library(viridis)
image <- image_read(img_path)
info <- image_info(image)
geometry <- sprintf("%dx%d!", info$width, info$height)
pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "daisy_overlay.png",
              width = 14, height = 14, bg = NA, col = pal_col)
image_read("daisy_overlay.png") %>%
  image_resize(geometry, filter = "quadratic") %>%
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()

######################## Visulaization ############################
###################################################################