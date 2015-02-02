library(plyr)
library(ggplot2)

dat = read.csv('data/train_decode.csv')
dat = transform(dat,
                area = width * height)
dat2 = ddply(dat, 'label', summarize, 
             area = mean(area))
dat2 = dat2[order(dat2$area), ]
dat$label = factor(dat$label, 
                   levels=as.character(dat2$label))

p1 = ggplot(data=dat) +
     geom_boxplot(aes(x=label, y=area)) + 
     scale_y_continuous(trans='log2')
ggsave('img/area.png', p1, width=9, height=3)

