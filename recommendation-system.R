
##############################################################################################
# HA W.M.AlEX        October 16,2024                                                         #
#                                                                                            #
# Harvard Capstone Project: Movie Recommendation System - cod Version 1.0                    #
#                                                                                            #
# Notes: 1) The System takes approximately 25 minutes to complete the whole process.         #
#                                                                                            #  
#        2) Run time can be less or more depend on the specification of your System/Machine. #
#                                                                                            #
##############################################################################################
options(warn=-1)

# Install Necessary Packages if required
#
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org" )
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(rstudioapi)) install.packages("rstudioapi", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(tinytex)){install.packages("tinytex", repos = "http://cran.us.r-project.org")
                      tinytex::install_tinytex()}


# Loading Necessary Libraries
#
library(dslabs)
library(tidyverse)
library(dplyr)
library(tidyr)
library(lubridate)
library(stringr)
library(ggplot2)
library(gridExtra)
library(knitr)
library(tinytex)
library(rstudioapi)
library(caret)


# Set number of significant digits=6 globally
#
options(digits=6)

set.seed(1990)


###########################################################################
#                                                                         #
#  Initialization: Movie Recommendation System                            #
#                                                                         #
#   - Download Movielens datasets                                         #
#                                                                         #
#   - Create edx and final_holdout_test sets using movielens datasets     #
#                                                                         #
###########################################################################


#
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
#

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
      unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
      unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
              mutate(userId = as.integer(userId),
              movieId = as.integer(movieId),
              rating = as.numeric(rating),
              timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
            mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")


#
# Final hold-out test set will be 10% of MovieLens data
#
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1)                       # if using R 3.5 or earlier

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


# Make sure userId and movieId in final hold-out test set are also in edx set
#
final_holdout_test <- temp %>% 
                       semi_join(edx, by = "movieId") %>%
                       semi_join(edx, by = "userId")


# Add rows removed from final hold-out test set back into edx set
#
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#####################################
#                                   #
# Data Explorations - edx datasets  #
#                                   #
#####################################


# Dimension of edx - Number of rows and columns
#
dim_df <- data.frame(Rows = dim(edx)[1], Columns = dim(edx)[2])
dim_df %>% knitr::kable(caption="Dimension of edx dataset")                   


# Column name and Class of edx 
#
class_results <- data.frame(Class=sapply(edx, class))
class_results %>% knitr::kable(caption="edx dataset")


# Compute Number of Unique movieId, userId and genres.
#
u_movie_n  <- length(unique(edx$movieId))      
u_user_n   <- length(unique(edx$userId))       
u_genres_n <- length(unique(edx$genres))     

um_result <- data.frame(Description="Number of Unique movieId", Count=u_movie_n )
um_result <- bind_rows(um_result,
                       data.frame(Description="Number of unique userId", Count=u_user_n))
um_result <- bind_rows(um_result,
                       data.frame(Description="Number of unique combined genres", Count=u_genres_n))
um_result %>% knitr::kable(caption="edx datset")


# Occurrences of rating (By movieId)
#
# List of 10 Examples - Counting the Occurrences of rating (By movieId)
#
m_result <- edx %>% count(movieId,name="Occurency") %>%
                     count(Occurency, name="Count") %>%
                     arrange(desc(Count)) %>%
                     dplyr::slice(1:10)
m_result %>% knitr::kable(caption="Counting Occurrences of rating (By movieId)")


# Occurrences of rating (By userId)
#
# List of 10 Examples - Counting the Occurrences of rating (By userId) 
#
u_result <- edx %>% count(userId,name="Occurency") %>%
                     count(Occurency, name="Count") %>%
                     dplyr::slice(30:40)
u_result %>% knitr::kable(caption="Counting Occurrences of rating (By userId)")


# ggplot histogram - m1 and m2 
#
m1 <- edx %>% 
          dplyr::count(movieId) %>% 
          ggplot(aes(n)) + 
          geom_histogram(bins = 50, fill="#006EBB", color="black") +
          labs(y="Count",x="Number of Occurrence") +
          scale_x_log10() +
          ggtitle("Movies")
m2 <- edx %>% 
          dplyr::count(userId) %>% 
          ggplot(aes(n)) + 
          geom_histogram(bins = 200, fill="cyan", color="#D883B7"   ) + 
          labs(y="Count",x="Number of Occurrence") +
          scale_x_log10()+ 
          ggtitle("Users")
grid.arrange(m1, m2, ncol = 2)


# List missing value (if any) in column rating of edx
#
na_results  <- edx[apply(is.na(edx),1,any),]
na_results %>% select(userId,movieId,rating,timestamp,title,genres) %>%
               knitr::kable(caption="Columns with NA values in edx dataset")


# List zeros value (if any) in column rating of edx
#
length(which(edx$rating==0))


# List of 8 Examples: the Original edx - rating given by one User to one Movie.
#
edx %>%
    group_by(title) %>%
    mutate(f=length(rating)) %>% ungroup() %>%
    filter(f > 10000) %>%
    distinct(title, .keep_all=TRUE) %>% 
    arrange(desc(f)) %>% 
    select(movieId,title,genres,userId,rating,timestamp) %>%
    dplyr::slice(1:8)


# Data wrangling - edx_original is retaining for Analysis 
#
edx_original <- edx


##################################################################
# R object and .rda files handling for current working directory #
##################################################################

# Specify the path relative to current working directory
#
edx_original_rda <- "rda/edx-original-rda.rda"

# Extract the directory path from the file path
#
dir_path <- dirname(edx_original_rda)
print(dir_path)

# Create the directory if it doesn't exist
#
if (!dir.exists(dir_path)) {
          dir.create(dir_path)
}

# save edx to edx-original-rda.rda file for working within R
#
save(edx_original, file=edx_original_rda)


###################################################################
# R Object and .rda files Handling for R script current directory #
###################################################################

# Get R script current directory
#
current_path_r_script <- rstudioapi::getActiveDocumentContext()$path
current_directory_r_script <- dirname(current_path_r_script)
print(current_directory_r_script)


# Join R script current directory path and file name
#
edx_original_rda <- str_c(current_directory_r_script,"/rda/edx-original-rda.rda")
print(edx_original_rda)


# Extract the directory path from the file path
#
dir_path <- dirname(edx_original_rda)
print(dir_path)


# Create the directory if it doesn't exist
#
if (!dir.exists(dir_path)) {
  dir.create(dir_path)
}


# save edx_original to edx-original-rda.rda file for working within R
#
save(edx_original, file=edx_original_rda)


# Mean of rating by genres BEFORE Combined genres Separation 
#
# List of 20 Examples - Mean of rating by genres in descending order by Number of rating. (count > 10000)
#
genres_avg <- edx %>%
                   group_by(genres) %>%
                   summarize(average_rating_genres=mean(rating, na.rm=TRUE ),
                   se=sd(rating, na.rm=TRUE)/sqrt(n()), count=n()) %>%
                   filter(count > 10000) %>%
                   arrange(desc(average_rating_genres)) %>%
                   mutate(rank=rownames(.))

genres_avg_plot <- genres_avg %>% dplyr::slice(1:20)

genres_avg %>% select(rank, genres, average_rating_genres, count) %>%
               dplyr::slice(1:20)


# Mean of rating by genres (genres="Drama")
#
genres_avg %>% select(rank, genres, average_rating_genres, count) %>%
               dplyr::slice(48)


# Mean of rating by genres (genres="Comedy")
#
genres_avg %>% select(rank, genres, average_rating_genres, count) %>%
               dplyr::slice(131)


# List of rank, Combined genres, Mean of rating by genres and Number of rating of "Comedy" and "Drama".
#
genres_t1 <- genres_avg %>%
                         filter(genres%in%c("Comedy","Drama")) %>%
                         select(rank, genres, average_rating_genres, count)

genres_t1 %>% knitr::kable(caption="BEFORE Combined genres Separation")


# ggplot Mean of rating by genres with error bars BEFORE genres Separation
#
genres_avg_plot %>%
                 ggplot(aes(x=genres, y=average_rating_genres)) +
                 geom_errorbar(aes(ymin=average_rating_genres - se,
                                   ymax=average_rating_genres + se),color="#7977B8") +
                 geom_point(color="#F89E78") +
                 theme(axis.text.x=element_text(angle=90, vjust=0.5, hjust=1)) +
                 labs(x="Combined genres", y="Mean of rating",
                 title="(Original Combined genres) Mean of rating by genres with Error Bars")


# Mean of rating by genres AFTER Combined genres Separation 
#
# List of 15 Examples - Mean of rating by genres in descending order. (count > 20000)
#
# Data Wrangling - Separate Combined genres into several Rows & each Row contains only one Genre.
#
genres_avg <- edx %>%
                  separate_longer_delim(genres, delim="|") %>%
                  group_by(genres) %>%
                  summarize(average_rating_genres=mean(rating, na.rm=TRUE ),
                  se=sd(rating, na.rm=TRUE)/sqrt(n()), count=n()) %>%
                  filter(count > 200000) %>%
                  arrange(desc(average_rating_genres)) %>%
                  mutate(rank=rownames(.))  

genres_avg_plot <- genres_avg %>% dplyr::slice(1:20)

genres_avg %>% select(rank, genres, average_rating_genres, count) %>%
               dplyr::slice(1:20)


# List of rank, Combined genres, Mean of rating by genres and Number of rating of "Comedy" and "Drama".
#
genres_t1 <- genres_avg %>%
                        filter(genres%in%c("Comedy","Drama")) %>%
                        select(rank, genres, average_rating_genres,count)
genres_t1 %>% knitr::kable(caption="After Separation of Combined genres")


# ggplot Mean of rating by genres with Error Bars AFTER genres Separated into several Rows
#
genres_avg_plot %>%
        ggplot(aes(x=genres, y=average_rating_genres)) +
        geom_errorbar(aes(ymin= average_rating_genres- se,
                          ymax= average_rating_genres+ se),color="#7977B8") +
        geom_point(color="#F89E78") +
        theme(axis.text.x=element_text(angle=90, vjust=0.5, hjust=1)) +
        labs(x="Separated genres", y="Mean of rating",
        title="(After Separation of genres) Mean of rating by genres with Error Bars") +
        scale_y_continuous(breaks=seq(0.5,5,by=0.1))


# Data Wrangling - generate new Columns (d_w, d_m, d_y) from column timestamp
#
edx <- edx %>% mutate(d_w=format(round_date(as_datetime(timestamp),"week"),"%Y-%m-%d"))
edx <- edx %>% mutate(d_m=format(round_date(as_datetime(timestamp),"month"),"%Y-%m-%d"))
edx <- edx %>% mutate(d_y=format(round_date(as_datetime(timestamp),"year"),"%Y-%m-%d"))


# Data Wrangling - generate new Columns (m_r, m_rw, m_ry, m_rg, tot_nr) from column rating
#
edx <- edx %>% group_by(movieId) %>% mutate(m_r = mean(rating, na.rm=TRUE))
edx <- edx %>% group_by(movieId,d_w) %>% mutate(m_rw= mean(rating, na.rm=TRUE))
edx <- edx %>% group_by(movieId,d_y) %>% mutate(m_ry= mean(rating, na.rm=TRUE))
edx <- edx %>% group_by(userId,genres) %>% mutate(m_rg=mean(rating, na.rm=TRUE))


# Data Wrangling - generate new Columns tot_nr
#
edx <- edx %>% group_by(movieId) %>% mutate(tot_nr=n()) %>% ungroup()


# Data Wrangling - generate new Column "release" from column title 
#
edx <- edx %>% mutate(release = str_extract(title,"\\d{4}")) %>%
               mutate(title = str_replace(title,"\\s*\\(\\d{4}\\)",""))


# Data Wrangling for ggplot. edx datasets is retaining for Analysis
#
edx <- edx %>% mutate(yday_dw=yday(d_w), month_dm=month(d_m), year_dy=year(d_y))


##################################################################
# R object and .rda files handling for current working directory #
##################################################################

# Specify the path relative to current working directory
#
edx_rda <- "rda/edx-rda.rda"

# Extract the directory path from the file path
#
dir_path <- dirname(edx_rda)
print(dir_path)

# Create the directory if it doesn't exist
#
if (!dir.exists(dir_path)) {
  dir.create(dir_path)
}

# save edx to edx-rda.rda file for working within R
#
save(edx, file=edx_rda)


###################################################################
# R object and .rda files handling for R script current directory #
###################################################################
 
# Get R script current directory
#
current_path_r_script <- rstudioapi::getActiveDocumentContext()$path
current_directory_r_script <- dirname(current_path_r_script)
print(current_directory_r_script)

# Concatenate R script current directory path and file name
#
edx_rda <- str_c(current_directory_r_script,"/rda/edx-rda.rda")
print(edx_rda)

# Extract the directory path from the file path
#
dir_path <- dirname(edx_rda)
print(dir_path)

# Create the directory if it doesn't exist
#
if (!dir.exists(dir_path)) {
          dir.create(dir_path)
}

# save edx to edx-rda.rda file for working within R
#
save(edx, file=edx_rda)


# List of 10 Examples encompassing new Features (d_w)
#
edx %>% group_by(title) %>%
        mutate(f=length(rating)) %>% ungroup() %>%
        filter(f > 15000) %>%
        arrange(desc(f),timestamp) %>% 
        distinct(title, .keep_all=TRUE) %>% 
        select(userId,movieId,title,d_w,rating) %>%
        dplyr::slice(10:20)


# List of 8 Examples encompassing new Features (release, m_r, m_rw, m_ry, m_rg, tot_nr)  
#
edx %>% group_by(title) %>%
        mutate(f=length(rating)) %>% ungroup() %>%
        filter(f > 20000) %>%
        arrange(desc(f),timestamp) %>% 
        distinct(title, .keep_all=TRUE) %>% 
        select(userId,movieId,release,title,rating,m_r,m_rw,m_ry,m_rg,tot_nr) %>%
        dplyr::slice(1:8)


# List of 20 Movies - most frequently Rated by User with frequency over 10000 (frequency > 10000)
#
c_1 <- edx %>%
            group_by(title) %>%
            mutate(frequency=length(rating)) %>% ungroup() %>%
            filter(frequency > 10000) %>%
            distinct(title, .keep_all=TRUE) %>% 
            arrange(desc(frequency)) %>%
            mutate(rank=rownames(.)) %>%
            select(rank,title,frequency) %>%
            dplyr::slice(1:20)

c_1 %>% knitr::kable(caption="Frequency of rating By title")


# ggplot - Frequency of rating By title
#
c_1 %>% mutate(title=reorder(title,frequency)) %>% 
        ggplot(aes(title,frequency)) + 
        geom_bar(stat="identity",fill="#D883B7",color="white")  +
        labs(y="Frequency",
             x="Moive title",title="Frequency of User rating By Movie title") +
        scale_y_continuous(breaks=seq(0,33000,by=3000)) +
        coord_flip()


# List of 20 genres - most frequently Rated by User with frequency over 10000. (Frequency > 10000)
#
c_2 <- edx %>%
           group_by(genres) %>%
           mutate(frequency=length(rating)) %>% ungroup() %>%
           filter(frequency > 10000) %>%
           distinct(genres, .keep_all=TRUE) %>% 
           arrange(desc(frequency)) %>%
           mutate(rank=rownames(.)) %>% 
           select(rank,genres,frequency) %>%
           dplyr::slice(1:20)

print_df <- function(title,df)
              {
                 cat(title, "\n\n")
                 cat(capture.output(print(n=50,df)), sep="\n")
              }
print_df("List of 20 genres - Frequency of rating By genres ",c_2)


# ggplot -  Frequency of rating By genres
#
c_2 %>% mutate(genres=reorder(genres,frequency)) %>%
        ggplot(aes(genres,frequency)) + 
        geom_bar(stat="identity",fill="#584298",color="white")  +
        labs(y="Frequency",x="genres",title="Frequency of User rating By genres") +
        scale_y_continuous(breaks=seq(0,800000,by=80000)) +
        coord_flip()


# ggplot - Correlation of Normalized Mean of rating and rating
#
r <- edx %>%
         summarize(r=cor(rating,m_r)) %>%
         pull(r)
r <- round(r,6)

edx %>% mutate(rating=scale(rating),m_r=scale(m_r)) %>%
        group_by(rating) %>%
        summarize(m_r=mean(m_r)) %>%
        ggplot(aes(rating,m_r)) + geom_point()+
        geom_abline(intercept=0, slope=r,color="#F89E78")+
        labs(x="scale(rating)",y="scale(m_r)",
        title=paste("Normalized Mean of rating vs rating [cor =",r,"]"))    


# ggplot - Correlation of Normalized Mean by Week of Date and rating
#
r <- edx %>%
         summarize(r=cor(rating,m_rw)) %>%
         pull(r)
r <- round(r,6)

edx %>% mutate(rating=scale(rating),m_rw=scale(m_rw)) %>%
        group_by(rating) %>%
        summarize(m_rw=mean(m_rw)) %>%
        ggplot(aes(rating,m_rw)) + geom_point()+
        geom_abline(intercept=0, slope=r,color="#F89E78")+
        labs(x="scale(rating)",y="scale(m_rw)",
        title=paste("Normalized Mean of rating by Week of Date vs rating [cor =",r,"]"))    


# ggplot - Correlation of Normalized Mean by Year of Date and rating
#
r <- edx %>%
         summarize(r=cor(rating,m_ry)) %>%
         pull(r)
r <- round(r,6)

edx %>% mutate(rating=scale(rating),m_ry=scale(m_ry)) %>%
        group_by(rating) %>%
        summarize(m_ry=mean(m_ry)) %>%
        ggplot(aes(rating,m_ry)) + geom_point()+
        geom_abline(intercept=0, slope=r,color="#F89E78")+
        labs(x="scale(rating)",y="scale(m_ry)",
        title=paste("Normalized Mean of rating by Year of Date vs rating [cor =",r,"]"))


# ggplot - Correlation of Normalized Mean of rating by genres and rating
#
r <- edx %>%
         summarize(r=cor(rating,m_rg)) %>%
         pull(r)
r <- round(r,6)

edx %>% mutate(rating=scale(rating),m_rg=scale(m_rg)) %>%
        group_by(rating) %>%
        summarize(m_rg=mean(m_rg)) %>%
        ggplot(aes(rating,m_rg)) + geom_point()+
        geom_abline(intercept=0, slope=r,color="#F89E78") +
        labs(x="scale(rating)",y="scale(m_rg)",
        title=paste("Normalized Mean of rating by genres vs rating [cor =",r,"]"))


# Correlation table - new Features (m_r, m_rw, m_ry, m_rg) of edx
#
avg_r_all <- edx %>%
                 select(rating,m_r,m_rw,m_ry,m_rg)
cor_r_all <- cor(na.omit(avg_r_all[, unlist(lapply(avg_r_all, is.numeric))]))
cor_r_all %>% knitr::kable(caption="Correlation - new Features of edx dataset")


# ggplot - rating Distributions of Movie
#
edx %>% ggplot(aes(rating)) +
        geom_histogram(binwidth=0.5,fill="#7977B8",color="black") +
        labs(y="Frequency",x="rating",title="rating Distributions of Movie") +
        scale_x_continuous(breaks=seq(0.5,5,by=0.5))

        
# ggplot - Distributions of Movie (Most Given rating in order from Most to Least)
#
edx_asc <- edx %>%
               group_by(rating) %>%
               summarise(Frequency = n()) %>%
               arrange(Frequency)

edx_asc %>% ggplot(aes(x=reorder(rating, Frequency), y=Frequency)) +
            geom_bar(stat="identity", width=0.5, aes(fill=Frequency)) +
            scale_fill_gradient(low="red", high="blue") +
            coord_flip() +
            labs(y="Frequency",
                 x="rating",
                 title = "Distributions of Movie (Most Given rating in order from Most to Least)") 
            scale_x_continuous(breaks = seq(0.5, 5, by = 0.5))

            
# ggplot - Mean of rating Distributions
#
edx %>% ggplot(aes(m_r)) +
        geom_histogram(binwidth=0.1,fill="#C6DC67",color="black") +
        labs(y="Frequency",x="Mean of rating",title="Mean of rating Distributions") +
        theme(axis.text.x=element_text(angle=90)) +
        scale_x_continuous(breaks=seq(0.5,5,by=0.1))


# ggplot - Mean of rating Distributions By movieId , Week of Date
#
edx %>% ggplot(aes(m_rw)) +
        geom_histogram(binwidth=0.1,fill="#C6DC67",color="black") +
        labs(y="Frequency",x="Mean of rating",
             title="Mean of rating Distributions By movieId, Week of Date") +
        theme(axis.text.x=element_text(angle=90)) +
        scale_x_continuous(breaks=seq(0.5,5,by=0.1))


# ggplot - Mean of rating Distributions By movieId , Year of Date
#
edx %>% ggplot(aes(m_ry)) +
        geom_histogram(binwidth=0.1,fill="#C6DC67",color="black") +
        labs(y="Frequency",x="Mean of rating",
             title="Mean of rating Distributions By movieId, Year of Date") +
        theme(axis.text.x=element_text(angle=90)) +
        scale_x_continuous(breaks=seq(0.5,5,by=0.1))


# ggplot - Mean of rating Distributions (By userId, genres)
#
edx %>% ggplot(aes(m_rg)) +
        geom_histogram(binwidth=0.1,fill="cyan",color="black") +
        labs(y="Frequency",x="Mean of rating",
             title="Mean of rating Distributions By userId, genres") +
        theme(axis.text.x=element_text(angle=90)) +
        scale_x_continuous(breaks=seq(0.5,5,by=0.1))


# ggplot - Number of rating Distributions By movieId
#
edx %>% ggplot(aes(movieId)) +
        geom_histogram(binwidth=1,color="#7977B8") +
        labs(y="Number of rating",x="movieId",
             title="Number of rating Distributions By movieId") +
        theme(axis.text.x=element_text(angle=90)) +
        scale_x_continuous(breaks=seq(0,70000,by=7000))


# ggplot - rating Distributions through the day of year
#
edx %>% ggplot(aes(yday_dw)) +   
        geom_histogram(binwidth=0.05,color="#F69289") +
        labs(y="Frequency",x="Day",
             title="rating Distributions through the Day of Year") +
        scale_x_continuous(breaks=seq(0,400,by=50))


# ggplot - rating Distributions (Month)
#
edx %>% ggplot(aes(month_dm)) + 
        geom_histogram(binwidth=1,fill="#46C5DD",color="black") +
        labs(y="Frequency",x="Months",title="rating Distributions (Month)") +
        scale_x_continuous(breaks=seq(1,12,by=1))


# ggplot - rating Distributions (Year)
#
edx %>% ggplot(aes(year_dy)) + 
        geom_histogram(binwidth=1,fill="#46C5DD",color="black") +
        labs(y="Frequency",x="Years",title="rating Distributions (Year)") +
        theme(axis.text.x=element_text(angle=90)) +
        scale_x_continuous(breaks=seq(1993,2020,by=1))


#######################################################################################
#       Models/Algorithm training of train_set and testing on test_set datasets       #
#                                                                                     #
#######################################################################################
#                                                                                     #
# 1. Split the edx datasets into separate train_set and test_set datasets.            #
#                                                                                     #
# 2. Train different Models/Algorithm on train_set datasets using Cross-Validation    #
#    and Regularization Method.                                                       #
#                                                                                     #
# 3. Determine Minimum Lambda value of Regularized Models during the process.         #
#                                                                                     #     
# 4. Model Testing on test_set datasets.                                              #
#                                                                                     #
# 5. Compute RMSE for each Model.                                                     #
#                                                                                     #
# 6. Retain Minimum Lambda values for using in Final Model Building.                  #
#                                                                                     #
#######################################################################################

#
# Partition and Create train_set (80% of edx datasets) & test_set (20% of edx datasets)   
#

set.seed(755)

test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)

train_set <- edx[-test_index,]

test_set  <- edx[test_index,]


#
# Make sure movieId and userId are exist in both train_set and test_set
#
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#
# RMSE Function
#
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}  


####################################################################
#  Model/Algorithm Training of train_set and Testing  on test_set  #
#                                                                  #
####################################################################

set.seed(1990)

#
# "Naive Mean Based Model" training of train_set and testing on test_set then compute RMSE. 
#

# Compute Mean of ratings of train_set
mu  <- mean(train_set$rating, na.rm=TRUE)

# Compute RMSE on test_set
rmse_naive <- RMSE(mu, test_set$rating)


#
# "Movie Effects Based Model" training of train_set and testing on test_set then compute RMSE. 
#

# Compute Mean of ratings of train_set
mu <- mean(train_set$rating,na.rm=TRUE ) 

# Compute b_i (Movie Effects) of train_set
movie_avgs <- train_set %>% 
                 group_by(movieId) %>% 
                 summarize(b_i = mean(rating - mu))

# Compute predicted ratings (Mean of ratings + Movie Effects)
predicted_ratings <- test_set %>% 
                left_join(movie_avgs, by='movieId') %>%
                mutate(pred = mu + b_i ) %>%
                pull(pred)

# Compute RMSE on test_set
rmse_m <- RMSE(predicted_ratings, test_set$rating)


#
# "Movie+User Effects Based Model" training of train_set and testing on test_set then compute RMSE.
#
set.seed(1)

# Compute b_i (Movie Effects) of train_set
movie_avgs <- train_set %>% 
                     group_by(movieId) %>% 
                     summarize(b_i = mean(rating - mu))

# Compute b_u (User Effects) of train_set
user_avgs <- train_set %>% 
                       left_join(movie_avgs, by='movieId') %>%
                       group_by(userId) %>%
                       summarize(b_u = mean(rating - mu - b_i))

# Compute predicted ratings (Mean of ratings + Movie Effects + User Effects)
predicted_ratings <- test_set %>% 
                       left_join(movie_avgs, by='movieId') %>%
                       left_join(user_avgs, by='userId') %>%
                       mutate(pred = mu + b_i + b_u ) %>%
                       pull(pred)

# Compute RMSE on test_set
rmse_mu <- RMSE(predicted_ratings, test_set$rating)


#
# "Movie+User+Genres Effects Based Model" training of train_set and testing on test_set then compute RMSE. 
#
set.seed(23)

# Compute Mean of ratings of train_set
mu <- mean(train_set$rating, na.rm=TRUE)

# Compute b_i (Movie Effects) of train_set
b_i <- train_set %>% 
                group_by(movieId) %>%
                summarize(b_i = mean(rating - mu))

# Compute b_u (User Effects) of train_set
b_u <- train_set %>% 
                left_join(b_i, by="movieId") %>%
                group_by(userId) %>%
                summarize(b_u = mean(rating - b_i - mu))

# Compute b_g (Genres Effects) of train_set
b_g <- train_set %>% 
                left_join(b_u, by="userId") %>%
                group_by(genres) %>%
                summarize(b_g = mean(rating - b_u - m_rg))

# Compute predicted ratings (Mean of ratings + Movie Effects + User Effects + Genres Effects)
predicted_ratings <-  test_set %>% 
                        left_join(b_i, by = "movieId") %>%
                        left_join(b_u, by = "userId")  %>%
                        left_join(b_g, by = "genres") %>%
                        mutate(pred = mu + b_i + b_u + b_g )  %>%
                        pull(pred)

# Compute RMSE on test_set
rmse_mug <-  RMSE(predicted_ratings, test_set$rating)


#
# "Movie+User+Time Effects Based Model" training of train_set and testing on test_set then compute RMSE.
#
set.seed(755)

# Compute Mean of ratings of train_set
mu <- mean(train_set$rating, na.rm=TRUE)

# Compute b_i (Movie Effects) of train_set
b_i <- train_set %>% 
                 group_by(movieId) %>%
                 summarize(b_i = mean(rating - mu))

# Compute b_u (User Effects) of train_set
b_u <- train_set %>% 
                 left_join(b_i, by="movieId") %>%
                 group_by(userId) %>%
                 summarize(b_u = mean(rating - b_i - mu))

# Compute b_t (Time Effects) of train_set
b_t <- train_set %>% 
                 left_join(b_u, by="userId") %>%
                 group_by(movieId) %>%
                 summarize(b_t = mean(rating - b_u - m_rw))

# Compute predicted ratings (Mean of ratings + Movie Effects + User Effects + Time Effects)
predicted_ratings <- test_set %>% 
                         left_join(b_i, by = "movieId") %>%
                         left_join(b_u, by = "userId")  %>%
                         left_join(b_t, by = "movieId") %>%
                         mutate(pred = mu + b_i + b_u + b_t )  %>%
                         pull(pred)

# Compute RMSE on test_set
rmse_mut <-  RMSE(predicted_ratings, test_set$rating)


#
# "Regularized Movie Effects Based Model" training of train_set using Regularization Method and testing on test_set 
#  Compute RMSE and Minimum Lambda using Cross Validation Method.
#

# Generate sequence of lambdas ranging from 0 to 10 with increment of 0.25
lambdas <- seq(0, 10, 0.25)

# Function that generate and return a list of RMSES using Cross Validation Method
#   1) Compute Mean of ratings of train_set
#   2) Compute b_i (Movie Effects) of train_set using Regularization Method 
#   3) Compute predicted ratings (Mean of ratings + Movie Effects)
#
rmses_rm_list <- sapply(lambdas, function(lambd){
                        
                        mu <- mean(train_set$rating,na.rm=TRUE )
                        
                        movie_reg_avgs <- train_set %>% 
                                             group_by(movieId) %>% 
                                             summarize(b_i = sum(rating - mu)/(n()+lambd), n_i = n())
                        
                        predicted_ratings <- test_set %>% 
                                               left_join(movie_reg_avgs, by='movieId') %>%
                                               mutate(pred = mu + b_i ) %>%
                                               pull(pred)
                        
return(RMSE(predicted_ratings, test_set$rating))
})

# Compute the Minimum Lambda value of "Regularized Movie Effects Based Model"
min_lambda_m <- lambdas[which.min(rmses_rm_list)]


#
# "Regularized Movie+User Effects Based Model" training of train_set using Regularization Method and
#  testing on test_set. Compute RMSE and Minimum Lambda using Cross Validation Method.
#

# Generate sequence of lambdas ranging from 0 to 10 with increment of 0.25
lambdas <- seq(0, 10, 0.25)

# Function that generate and return a list of RMSES using Cross Validation Method
#   1) Compute Mean of ratings of train_set
#   2) Compute b_i (Movie Effects) of train_set using Regularization Method
#   3) Compute b_u (User Effects) of train_set using Regularization Method
#   4) Compute predicted ratings (Mean of ratings + Movie Effects + User Effects)
#
rmses_iu_list <- sapply(lambdas, function(lambd){

                          mu <- mean(train_set$rating, na.rm=TRUE)
                          
                          b_i <- train_set %>% 
                                 group_by(movieId) %>%
                                 summarize(b_i = sum(rating - mu)/(n()+lambd))
                          
                          b_u <- train_set %>% 
                                left_join(b_i, by="movieId") %>%
                                group_by(userId) %>%
                                summarize(b_u = sum(rating - b_i - mu)/(n()+lambd))
                          
                         predicted_ratings <- test_set %>% 
                                left_join(b_i, by = "movieId") %>%
                                left_join(b_u, by = "userId") %>%
                                mutate(pred = mu + b_i + b_u ) %>%     
                                pull(pred)
                         
return(RMSE(predicted_ratings, test_set$rating))
})

# Compute the Minimum Lambda value of "Regularized Movie+User Effects Based Model"
min_lambda_mu  <- lambdas[which.min(rmses_iu_list)]


#
# "Regularized Movie+User+Genres Effects Model" training of train_set using Regularization Method and
#  testing on test_set. Compute RMSE and Minimum Lambda using Cross Validation Method. 
#

# Generate sequence of lambdas ranging from 0 to 10 with increment of 0.25
lambdas <- seq(0, 10, 0.25)

# Function that generate and return a list of RMSES using Cross Validation Method
#   1) Compute Mean of ratings of train_set
#   2) Compute b_i (Movie Effects) of train_set using Regularization Method
#   3) Compute b_u (User Effects) of train_set using  Regularization Method
#   4) Compute b_g (Genres Effects) of train_set using Regularization Method
#   5) Compute predicted ratings (Mean of ratings + Movie Effects + User Effects + Genres Effects)
#
rmses_rmug_list <- sapply(lambdas, function(lambd){
  
                          mu <- mean(train_set$rating, na.rm=TRUE)
                          
                          b_i <- train_set %>% 
                                    group_by(movieId) %>%
                                    summarize(b_i = sum(rating - mu)/(n()+lambd))
                          
                          b_u <- train_set %>% 
                                    left_join(b_i, by="movieId") %>%
                                    group_by(userId) %>%
                                    summarize(b_u = sum(rating - b_i - mu)/(n()+lambd))
                                    
                          b_g <- train_set %>% 
                                    left_join(b_u, by="userId") %>%
                                    group_by(genres) %>%
                                    summarize(b_g = sum(rating - b_u - m_rg)/(n()+lambd))
                         
                         predicted_ratings <- test_set %>% 
                                                left_join(b_i, by = "movieId") %>%
                                                left_join(b_u, by = "userId")  %>%
                                                left_join(b_g, by = "genres") %>%
                                                mutate(pred = mu + b_i + b_u + b_g )  %>%
                                                pull(pred)
                         
return(RMSE(predicted_ratings, test_set$rating))
})

# Compute the Minimum Lambda value of "Regularized Movie+User+Genres Effects Based Model"
min_lambda_mug  <- lambdas[which.min(rmses_rmug_list)]


#
# "Regularized Movie+User+Time Effects Based Model" training of train_set using Regularization Method and
#  testing on test_set. Compute RMSE and Minimum Lambda using Cross Validation Method.
#

# Generate sequence of lambdas ranging from 0 to 10 with increment of 0.25
lambdas <- seq(0, 10, 0.25)

# Function that generate and return a list of RMSES using Cross Validation Method
#   1) Compute Mean of ratings of train_set
#   2) Compute b_i (Movie Effects) of train_set using Regularization Method
#   3) Compute b_u (User Effects) of train_set using Regularization Method
#   4) Compute b_t (Time Effects) of train_set using Regularization Method
#   5) Compute predicted ratings (Mean of ratings + Movie Effects + User Effects + Time Effects)
#
rmses_rmut_list <- sapply(lambdas, function(lambd){
  
                          mu <- mean(train_set$rating, na.rm=TRUE)
                          
                          b_i <- train_set %>% 
                                      group_by(movieId) %>%
                                      summarize(b_i = sum(rating - mu)/(n()+lambd))
                          
                          b_u <- train_set %>% 
                                      left_join(b_i, by="movieId") %>%
                                      group_by(userId) %>%
                                      summarize(b_u = sum(rating - b_i - mu)/(n()+lambd))
                          
                          b_t <- train_set %>% 
                                      left_join(b_u, by="userId") %>%
                                      group_by(movieId) %>%
                                      summarize(b_t = sum( rating - b_u - m_rw ) /(n()+lambd))
                          
                          predicted_ratings <- test_set %>% 
                                      left_join(b_i, by = "movieId") %>%
                                      left_join(b_u, by = "userId")  %>%
                                      left_join(b_t, by = "movieId") %>%
                                      mutate(pred = mu + b_i + b_u + b_t )  %>%
                                      pull(pred)
                          
return(RMSE(predicted_ratings, test_set$rating))
})

# Compute the Minimum Lambda value of "Regularized Movie+User+Time Effects Based Model"
min_lambda_mut <- lambdas[which.min(rmses_rmut_list)]




#
# Data Wrangling - min_lambda is retaining for Analysis
#
min_lambda <- data_frame(movie             = min_lambda_m,
                         movie_user        = min_lambda_mu,
                         movie_user_genres = min_lambda_mug,
                         movie_user_time   = min_lambda_mut,
                         rmses_rm          = rmses_rm_list,
                         rmses_riu         = rmses_iu_list,
                         rmses_rmug        = rmses_rmug_list,
                         rmses_rmut        = rmses_rmut_list)


###################################################################
# R object and .rda files handling for current working directory  #
###################################################################

# Specify the path relative to current working directory
#
lambda_rda <- "rda/min-lambda.rda"
fht_rda <- "rda/final-holdout-test.rda"

# Extract the directory path from the file path
#
dir_path <- dirname(lambda_rda)
print(dir_path)


# Create the directory if it doesn't exist
#
if (!dir.exists(dir_path)) {
  dir.create(dir_path)
}

# save min_lambda to min-lambda.rda file for working within R
#
save(min_lambda, file=lambda_rda)


#
# Data Wrangling - final-holdout-test.rda for Rmd report
#

# Specify the path relative to current working directory
#
fht_rda <- "rda/final-holdout-test.rda"

# Extract the directory path from the file path
#
dir_path <- dirname(fht_rda)
print(dir_path)

# Create the directory if it doesn't exist
#
if (!dir.exists(dir_path)) {
  dir.create(dir_path)
}

# save final_holdout_test to final-holdout-test.rda file for working within R
#
save(final_holdout_test, file=fht_rda)


####################################################################
# R object and .rda files handling for R script current directory  #
####################################################################

# Get R script current directory
#
current_path_r_script <- rstudioapi::getActiveDocumentContext()$path
current_directory_r_script <- dirname(current_path_r_script)
print(current_directory_r_script)

# Concatenate R script current directory path and file name
#
lambda_rda <- str_c(current_directory_r_script,"/rda/min-lambda.rda")
print(lambda_rda)

# Extract the directory path from the file path
#
dir_path <- dirname(lambda_rda)
print(dir_path)

# Create the directory if it doesn't exist
#
if (!dir.exists(dir_path)) {
         dir.create(dir_path)
}

# save min_lambda to min-lambda.rda file for working within R
#
save(min_lambda, file=lambda_rda)


#
# Data Wrangling - final-holdout-test.rda for Rmd report
#

# Get R script current directory
#
current_path_r_script <- rstudioapi::getActiveDocumentContext()$path
current_directory_r_script <- dirname(current_path_r_script)
print(current_directory_r_script)

# Concatenate R script current directory path and file name
#
fht_rda <- str_c(current_directory_r_script,"/rda/final-holdout-test.rda")
print(fht_rda)

# Extract the directory path from the file path
#
dir_path <- dirname(fht_rda)
print(dir_path)

# Create the directory if it doesn't exist
#
if (!dir.exists(dir_path)) {
  dir.create(dir_path)
}

# save final_holdout_test to final-holdout-test.rda file for working within R
#
save(final_holdout_test, file=fht_rda)



######################################################################################
#     Build Final Models - Training of edx and testing on final_holdout_test sets    #
#                                                                                    #
######################################################################################
#                                                                                    #
# 1. Models Training of edx datasets                                       .         #
#                                                                                    #
# 2. Apply relevant Minimum Lambda value.                                             #
#                                                                                    #     
# 3. Final Models Testing on final_holdout_test sets.                                #
#                                                                                    #
# 4. Compute RMSE for Final Models.                                                  #
#                                                                                    #
# 5. Generate Final Model RMSE table.                                                #
#                                                                                    #
######################################################################################

#
# Build "Naive Mean Based Model"- Training of edx and Testing on final_holdout_test sets.
#                                 compute Final Model RMSE.
#
set.seed(755)

# Compute Mean of ratings using edx
mu  <- mean(edx$rating, na.rm=TRUE)

# Compute RMSE on final_holdout_test
final_model_rmse_naive <- RMSE(mu, final_holdout_test$rating)

# Generate "Naive Mean Based Model" Final Model RMSE table
final_model_rmse_table  <- data_frame(MODEL = "Naive Mean Based Model", RMSE = final_model_rmse_naive)


#
# Build "Movie Effects Based Model"- Training of edx and Testing on final_holdout_test sets.
#                                    compute Final Model RMSE.
#

# Compute Mean of ratings of edx
mu <- mean(edx$rating,na.rm=TRUE ) 

# Compute b_i (Movie Effects) of edx
movie_avgs <- edx %>% 
                  group_by(movieId) %>% 
                  summarize(b_i = mean(rating - mu))

# Compute predicted ratings (Mean of ratings + Movie Effects)
predicted_ratings <- final_holdout_test %>% 
                          left_join(movie_avgs, by='movieId') %>%
                          mutate(pred = mu + b_i ) %>%
                          pull(pred)

# Compute RMSE on final_holdout_test
final_model_rmse_m <- RMSE(predicted_ratings, final_holdout_test$rating)

# Generate "Movie Effects Based Model" Final Models RMSE table
final_model_rmse_table <- bind_rows(final_model_rmse_table,
                          data_frame(MODEL ="Movie Effects Based Model",
                                     RMSE = final_model_rmse_m ))


# qplot Histogram- Movie Effects (frequency vs b_i)
#
movie_avgs %>%
            qplot(b_i,geom ="histogram",bins=30,data =.,color=I("black")) +
            labs(title="Movie Effects Histogram (b_i)")


# ggplot Histogram - User Effects (frequency vs b_u)
#
edx %>%
     group_by(userId) %>% 
     summarize(b_u = mean(rating)) %>% 
     ggplot(aes(b_u)) + 
     geom_histogram(bins = 30, color = "black") +
     labs(title="User Effects Histogram (b_u)")


#
# Build "Movie+User Effects Based Model"- Training of edx and Testing final_holdout_test sets.
#                                         compute Final Model RMSE.
#

# Compute Mean of ratings of edx
mu <- mean(edx$rating, na.rm=TRUE)

# Compute b_i (Movie Effects) of edx
movie_avgs <- edx %>% 
                group_by(movieId) %>% 
                summarize(b_i = mean(rating - mu))

# Compute b_u (User Effects) of edx
user_avgs <- edx %>% 
                 left_join(movie_avgs, by='movieId') %>%
                 group_by(userId) %>%
                 summarize(b_u = mean(rating - mu - b_i))

# Compute predicted ratings (Mean of ratings + Movie Effects + User Effects)
predicted_ratings <- final_holdout_test %>% 
                          left_join(movie_avgs, by='movieId') %>%
                          left_join(user_avgs, by='userId') %>%
                          mutate(pred = mu + b_i + b_u ) %>%
                          pull(pred)

# Compute RMSE on final_holdout_test
final_model_rmse_mu <- RMSE(predicted_ratings,final_holdout_test$rating)

# Generate "Movie+User Effects Based Model" Final Models RMSE table
final_model_rmse_table <- bind_rows(final_model_rmse_table,                       
                          data_frame(MODEL="Movie+User Effects Based Model",  
                                     RMSE = final_model_rmse_mu))


#
# Build "Movie+User+Genres Effects Based Model"- Training of edx and Testing on final_holdout_test sets.
#                                                Compute Final Model RMSE.
#

# Compute Mean of ratings of edx
mu <- mean(edx$rating, na.rm=TRUE)

# Compute b_i (Movie Effects) of edx
b_i <- edx %>% 
           group_by(movieId) %>%
           summarize(b_i = mean(rating - mu))

# Compute b_u (User Effects) of edx
b_u <- edx %>% 
           left_join(b_i, by="movieId") %>%
           group_by(userId) %>%
           summarize(b_u = mean(rating - b_i - mu))

# Compute b_g (Genres Effects) of edx
b_g <- edx %>% 
           left_join(b_u, by="userId") %>%
           group_by(genres) %>%
           summarize(b_g = mean(rating - b_u - m_rg))

# Compute predicted ratings (Mean of ratings + Movie Effects + User Effects + Genres Effects)
predicted_ratings <-  final_holdout_test %>% 
             left_join(b_i, by = "movieId") %>%
             left_join(b_u, by = "userId")  %>%
             left_join(b_g, by = "genres") %>%
             mutate(pred = mu + b_i + b_u + b_g )  %>%
             pull(pred)

# Compute RMSE on final_holdout_test
final_model_rmse_mug <-  RMSE(predicted_ratings, final_holdout_test$rating)

# Generate "Movie+User+Genres Effects Based Model" Final Models RMSE table
final_model_rmse_table <- bind_rows(final_model_rmse_table,                         
                          data_frame(MODEL ="Movie+User+Genres Effects Based Model",  
                                     RMSE = final_model_rmse_mug))


#
# Build "Movie+User+Time Effects Based Model"- Training of edx and Testing of final_holdout_test sets.
#                                              Compute Final Model RMSE.
#

# Compute Mean of ratings of edx
mu <- mean(edx$rating, na.rm=TRUE)

# Compute b_i (Movie Effects) of edx
b_i <- edx %>% 
           group_by(movieId) %>%
           summarize(b_i = mean(rating - mu))

# Compute b_u (User Effects) of edx
b_u <- edx %>% 
           left_join(b_i, by="movieId") %>%
           group_by(userId) %>%
           summarize(b_u = mean(rating - b_i - mu))

# Compute b_t (Time Effects) of edx
b_t <- edx %>% 
           left_join(b_u, by="userId") %>%
           group_by(movieId) %>%
           summarize(b_t = mean(rating - b_u - m_rw))

# Compute predicted ratings (Mean of ratings + Movie Effects + User Effects + Time Effects)
predicted_ratings <- final_holdout_test %>% 
             left_join(b_i, by = "movieId") %>%
             left_join(b_u, by = "userId")  %>%
             left_join(b_t, by = "movieId") %>%
             mutate(pred = mu + b_i + b_u + b_t )  %>%
             pull(pred)

# Compute RMSE on final_holdout_test
final_model_rmse_mut <-  RMSE(predicted_ratings, final_holdout_test$rating)

# Generate "Movie+User+Time Effects Based Model" Final Models RMSE table
final_model_rmse_table <- bind_rows(final_model_rmse_table,                          
                          data_frame(MODEL ="Movie+User+Time Effects Based Model",  
                                     RMSE = final_model_rmse_mut))


#
# Build "Regularized Movie Effects Based Model"- Training of edx datasets and
#                                                Testing on final_holdout_test sets.
# Apply the relevant Minimum Lambda value on the Algorithm and compute Final Model RMSE.
#

# Set lambd = Minimum lambda value of "Regularized Movie Effects Based Model"
lambd <- min_lambda_m

# Compute Mean of ratings of edx
mu <- mean(edx$rating,na.rm=TRUE )

# Compute b_i (Movie Effects) of edx datasets
movie_reg_avgs <- edx %>% 
                      group_by(movieId) %>% 
                      summarize(b_i = sum(rating - mu)/(n()+lambd), n_i = n())

# Compute predicted ratings (Mean of ratings + Movie Effects)
predicted_ratings <- final_holdout_test %>% 
                           left_join(movie_reg_avgs, by='movieId') %>%
                           mutate(pred = mu + b_i ) %>%
                           pull(pred)

# Compute RMSE on final_holdout_test
final_model_rmse_rm <-  RMSE(predicted_ratings, final_holdout_test$rating)

# Generate "Regularized Movie Effects Based Model" Final Models RMSE table
final_model_rmse_table <- bind_rows(final_model_rmse_table,
                          data_frame(MODEL ="Regularized Movie Effects Based Model",  
                                     RMSE = final_model_rmse_rm))

# qplot - Regularized Movie Effects (RMSES vs Lambdas)
#
qplot(lambdas, rmses_rm_list) +
  labs(x="Lambdas",y="RMSES",title="Regularized Movie Effects (RMSES vs Lambdas)")


#
# Build "Regularized Movie+User Effects Based Model"- Training of edx datasets and
#                                                     Testing on final_holdout_test sets.
# Apply the relevant Minimum Lambda value on Algorithm and compute Final Model RMSE.
#

# Set lambd = Minimum lambda value of "Regularized Movie+User Effects Based Model"
lambd <- min_lambda_mu

# Compute Mean of ratings of edx datasets
mu <- mean(edx$rating, na.rm=TRUE)

# Compute b_i (Movie Effects) of edx datasets
b_i <- edx %>%
           group_by(movieId) %>%
           summarize(b_i = sum(rating - mu) /(n()+lambd))

# Compute b_u (User Effects) of edx datasets
b_u <- edx %>% 
          left_join(b_i, by="movieId") %>%
          group_by(userId) %>%
          summarize(b_u = sum(rating - b_i - mu) /(n()+lambd))

# Compute predicted ratings (Mean of ratings + Movie Effects + User Effects)
predicted_ratings <- final_holdout_test %>% 
            left_join(b_i, by = "movieId") %>%
            left_join(b_u, by = "userId") %>%
            mutate(pred = mu + b_i + b_u ) %>%
            pull(pred)

# Compute RMSE on final_holdout_test
final_model_rmse_rmu <- RMSE(predicted_ratings, final_holdout_test$rating)

# Generate "Regularized Movie+User Effects Based Model" Final Models RMSE table
final_model_rmse_table <- bind_rows(final_model_rmse_table,
                          data_frame(MODEL = "Regularized Movie+User Effects Based Model",  
                                     RMSE  = final_model_rmse_rmu))

# qplot - Regularized Movie+User Effects (RMSES vs Lambdas)
#
qplot(lambdas, rmses_iu_list) + 
  labs(y="RMSES",x="Lambdas",title="Regularized Movie+User Effects (RMSES vs Lambdas)")


#
# Build "Regularized Movie+User+Genres Effects Based Model"- Training of edx datasets and
#                                                            Testing on final_holdout_test sets.
# Apply the relevant Minimum Lambda value on Algorithm and compute Final Model RMSE. 
#

# Set lambd = Minimum lambda value of "Regularized Movie+User+Genres Effects Based Model"
lambd <- min_lambda_mug

# Compute Mean of ratings of edx datasets
mu <- mean(edx$rating, na.rm=TRUE)

# Compute b_i (Movie Effects) of edx datasets
b_i <- edx %>% 
           group_by(movieId) %>%
           summarize(b_i = sum(rating - mu)/(n()+lambd))

# Compute b_u (User Effects) of edx datasets
b_u <- edx %>% 
          left_join(b_i, by="movieId") %>%
          group_by(userId) %>%
          summarize(b_u = sum(rating - b_i - mu)/(n()+lambd))

# Compute b_g (Genres Effects) of edx datasets
b_g <- edx %>% 
         left_join(b_u, by="userId") %>%
         group_by(genres) %>%
         summarize(b_g = sum(rating - b_u - m_rg)/(n()+lambd))

# Compute predicted ratings (Mean of ratings + Movie Effects + User Effects + Genres Effects)
predicted_ratings <-  final_holdout_test %>% 
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId")  %>%
        left_join(b_g, by = "genres") %>%
        mutate(pred = mu + b_i + b_u + b_g ) %>%
        pull(pred)

# Compute RMSE on final_holdout_test
final_model_rmse_rmug <- RMSE(predicted_ratings, final_holdout_test$rating)

# Generate "Regularized Movie+User+Genres Effects Based Model" Final Models RMSE table
final_model_rmse_table <- bind_rows(final_model_rmse_table,
                          data_frame(MODEL = "Regularized Movie+User+Genres Effects Based Model",  
                                     RMSE  = final_model_rmse_rmug))

# qplot - Regularized Movie+User+Genres Effects (RMSES vs Lambdas)
#
qplot(lambdas, rmses_rmug_list) + 
  labs(y="RMSES",x="Lambdas",title="Regularized Movie+User+Genres Effects (RMSES vs Lambdas)") 


#
# Build "Regularized Movie+User+Time Effects Based Model"- Training of edx datasets and
#                                                          Testing on final_holdout_test sets.
# Apply the relevant Minimum Lambda value on Algorithm and compute Final Model RMSE. 
#

# Set lambd = Minimum lambda value of "Regularized Movie+User+Time Effects Based Model"
lambd <- min_lambda_mut

# Compute Mean of ratings of edx datasets
mu <- mean(edx$rating, na.rm=TRUE)

# Compute b_i (Movie Effects) of edx datasets
b_i <- edx %>% 
           group_by(movieId) %>%
           summarize(b_i = sum(rating - mu)/(n()+lambd))

# Compute b_u (User Effects) of edx datasets
b_u <- edx %>% 
           left_join(b_i, by="movieId") %>%
           group_by(userId) %>%
           summarize(b_u = sum(rating - b_i - mu)/(n()+lambd))

# Compute b_t (Time Effects) of edx datasets
b_t <- edx %>% 
           left_join(b_u, by="userId") %>%
           group_by(movieId) %>%
           summarize(b_t = sum( rating - b_u - m_rw ) /(n()+lambd))

# Compute predicted ratings (Mean of ratings + Movie Effects + User Effects + Time Effects)
predicted_ratings <-  final_holdout_test %>% 
          left_join(b_i, by = "movieId") %>%
          left_join(b_u, by = "userId")  %>%
          left_join(b_t, by = "movieId") %>%
          mutate(pred = mu + b_i + b_u + b_t ) %>%
          pull(pred)

# Compute RMSE on final_holdout_test
final_model_rmse_rmut <- RMSE(predicted_ratings, final_holdout_test$rating)

# Generate "Regularized Movie+User+Time Effects Based Model" Final Models RMSE table
final_model_rmse_table <- bind_rows(final_model_rmse_table,
                          data_frame(MODEL = "Regularized Movie+User+Time Effects Based Model",  
                                     RMSE  = final_model_rmse_rmut))

# qplot - Regularized Movie+User+Time Effects (RMSES vs Lambdas)
#
qplot(lambdas, rmses_rmut_list) + 
  labs(y="RMSES",x="Lambdas",title="Regularized Movie+User+Time Effects (RMSES vs Lambdas)")


#####################################
#                                   #
#  Summary results of Final Models  #
#                                   #
#####################################


# Summary Table of Lambda values that give the Minimum RMSE for Regularized Final Models.
#
min_lambda_result <- data.frame(Effects = "Regularized Movie Effects", Lambdas = min_lambda_m )
min_lambda_result <- bind_rows(min_lambda_result,
                               data.frame(Effects="Regularized Movie+User Effects",Lambdas=min_lambda_mu))
min_lambda_result <- bind_rows(min_lambda_result,
                               data.frame(Effects="Regularized Movie+User+Genres Effects",Lambdas=min_lambda_mug))
min_lambda_result <- bind_rows(min_lambda_result,
                               data.frame(Effects="Regularized Movie+User+Time Effects",Lambdas=min_lambda_mut))
min_lambda_result %>% knitr::kable(caption="Lambdas give the Minimun RMSE")


#
# Summary Table of RMSES of Final Models/Algorithm
#
final_model_rmse_table %>% knitr::kable(caption="Result of Final Models - Root Mean Squared Error (RMSE)")


#
# Table of Final Model with Lowest RMSE
#
lowest_rmse_model <- data_frame(MODEL=final_model_rmse_table$MODEL[which.min(final_model_rmse_table$RMSE)],
                                RMSE =final_model_rmse_table$RMSE[which.min(final_model_rmse_table$RMSE)])
lowest_rmse_model %>% knitr::kable(caption="Final Model with Lowest RMSE")


####################################################################
#  Harvard Capstone Project: Movie Recommendation System - Ending  #
####################################################################
