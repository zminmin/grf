library(causalTree)
library(LaplacesDemon)
library(dplyr)
library(here)

# generate the CI for a forest
library(randomForestCI)

# for getting node prediction
library(partykit)
# for parallel runnning
# library(foreach)
# library(doParallel)
# for plotting result
# library(plotly)

library(grf)


# get data generation function
source(gsub("grf/r-package/grf", "MDTree/test/data_gen.R", here()))


this_type <- 0		# X binary 0, normal 1
this_design <- 2	# coefficient structure
this_m <- 4			  # X number when X is binary, structure when X is normal
# Attention: Xnumber can or cannot be the same to the variable "this_m"
if(this_type == 0) { Xnumber <- this_m }
if(this_type == 1) { Xnumber <- 4 }


train_size <- c(3000, 5000, 10000)	# train data size
forest_size <- 10	# forest number
boot_size <- 100	# tree number in one forest



grf_forest <- list()
for(for_index in 1:forest_size) {
	record <- list()
	record_index <- 1
	data_te <- data_build_new_version(10000, this_m, this_design, this_type, for_index)

	for (n in train_size) {
		dt <- data_build_new_version(n, this_m, this_design, this_type, for_index)

		print(paste0("forest ", for_index, " train size ", n))
		print("aggregate")
		forest <- custom_forest(dt[,1:Xnumber], dt$Y, dt$e1, dt$e2, dt$e3, dt$f1, dt$f2, dt$f3,
					ll.split.cutoff = 100, num.trees = boot_size, ci.group.size = 10, min.node.size = 150)
		pred = predict(forest, as.matrix(sapply(data_te[,1:Xnumber], as.numeric)),
					estimate.variance = TRUE)
		print("separate")
		c.forest.f1 = causal_forest(as.matrix(sapply(dt[,1:Xnumber], as.numeric)), dt$Y, dt$f1,
					num.trees = boot_size, honesty = TRUE, min.node.size = 150)
		pred.f1 = predict(c.forest.f1, as.matrix(sapply(data_te[,1:Xnumber], as.numeric)),
					estimate.variance = TRUE)
		c.forest.f2 = causal_forest(as.matrix(sapply(dt[,1:Xnumber], as.numeric)), dt$Y, dt$f2,
					num.trees = boot_size, honesty = TRUE, min.node.size = 150)
		pred.f2 = predict(c.forest.f2, as.matrix(sapply(data_te[,1:Xnumber], as.numeric)),
					estimate.variance = TRUE)
		c.forest.f3 = causal_forest(as.matrix(sapply(dt[,1:Xnumber], as.numeric)), dt$Y, dt$f3,
					num.trees = boot_size, honesty = TRUE, min.node.size = 150)
		pred.f3 = predict(c.forest.f3, as.matrix(sapply(data_te[,1:Xnumber], as.numeric)),
					estimate.variance = TRUE)


		temp_mse <- c()
		temp_cover_prob <- c()
		real_te = Get_real(data_te)
		for (i in 1:3) {
			mse <- mean((pred[,i] - real_te[,i+3])^2)
			cover_prob <- mean(real_te[,i+3]<=pred[,i] + 1.96*sqrt(pred[,i+3]) &
				real_te[, i+3]>=pred[,i] - 1.96*sqrt(pred[,i+3]))
			print(paste0(signif(mse, digits = 4), "   ", signif(cover_prob, digits = 4)))
			temp_mse[i] <- mse
			temp_cover_prob[i] <- cover_prob
		}

		temp_mse[4] <- mean((pred.f1$predictions - real_te[,4])^2)
		temp_mse[5] <- mean((pred.f2$predictions - real_te[,5])^2)
		temp_mse[6] <- mean((pred.f3$predictions - real_te[,6])^2)
		temp_cover_prob[4] <- mean(real_te[,4]<=pred.f1[,1] + 1.96*sqrt(pred.f1[,2]) &
				real_te[, 4]>=pred.f1[,1] - 1.96*sqrt(pred.f1[,2]))
		temp_cover_prob[5] <- mean(real_te[,5]<=pred.f2[,1] + 1.96*sqrt(pred.f2[,2]) &
				real_te[, 5]>=pred.f2[,1] - 1.96*sqrt(pred.f2[,2]))
		temp_cover_prob[6] <- mean(real_te[,6]<=pred.f3[,1] + 1.96*sqrt(pred.f3[,2]) &
				real_te[, 6]>=pred.f3[,1] - 1.96*sqrt(pred.f3[,2]))


		record[[record_index]] = list(MSE = temp_mse, Cover_Prob = temp_cover_prob, n_size = n)
		record_index = record_index + 1
	}

	grf_forest[[for_index]] = record
}

print("End Test ... ")

grf_AVE <- matrix(0, 6, length(train_size))
grf_COV_P <- matrix(0, 6, length(train_size))
for (tree_k in grf_forest) {
	for (k in 1:length(train_size)) {
		# print(k)
		# print(as.matrix(tree_k[[k]]$MSE))
		grf_AVE[, k] = grf_AVE[, k] + as.matrix(tree_k[[k]]$MSE)
		grf_COV_P[, k] = grf_COV_P[, k] + as.matrix(tree_k[[k]]$Cover_Prob)
	}
}
grf_AVE = grf_AVE / length(grf_forest)
grf_COV_P = grf_COV_P / length(grf_forest)

print(grf_AVE)
print(grf_COV_P)

Name_start <- "other_data_build_"
if(this_type == 0){
	if(this_m == 4){
		file_name <- paste0(Name_start, this_design, "_", 0, ".RData")}
	if(this_m == 6){
		file_name <- paste0(Name_start, this_design, "_", 1, ".RData")}
	if(this_m == 10){
		file_name <- paste0(Name_start, this_design, "_", 2, ".RData")}
	if(this_m == 15){
		file_name <- paste0(Name_start, this_design, ".RData")}
	if(this_m == 16){
		file_name <- paste0(Name_start, this_design, ".RData")}
}
if(this_type == 1){
	if(this_m == 4){
		file_name <- paste0(Name_start, this_design, "_", 0, "_N", ".RData")}
	if(this_m == 6){
		file_name <- paste0(Name_start, this_design, "_", 1, "_N", ".RData")}
	if(this_m == 10){
		file_name <- paste0(Name_start, this_design, "_", 2, "_N", ".RData")}
	if(this_m == 15){
		file_name <- paste0(Name_start, this_design, "_N", ".RData")}
	if(this_m == 16){
		file_name <- paste0(Name_start, this_design, "_N", ".RData")}
}

save(grf_AVE, grf_COV_P, grf_forest, file = file_name)