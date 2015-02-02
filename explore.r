library(plyr)
library(ggplot2)

dat = read.csv('data/train_decode.csv')
dat2 = ddply(dat, 'label', summarize, 
             area = mean(area))
dat2 = dat2[order(dat2$area), ]
dat$label = factor(dat$label, 
                   levels=as.character(dat2$label))

p1 = ggplot(data=dat) + theme_light() + 
     geom_point(aes(x=label, y=area), alpha=0.25) 
ggsave('img/area.png', p1, width=9, height=3)

