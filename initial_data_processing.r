set.seed(1)
train_dir = 'data/train'
tmp = dir(train_dir, recursive=F)
labs = 1:length(tmp)
names(labs) = tmp

imgs = dir(train_dir, recursive=T)

dat = data.frame(label = labs[gsub('/[a-z0-9]*.jpg', '', imgs, ignore.case=T)], 
                 fname = imgs, 
                 stringsAsFactors = F)
dat = dat[order(rnorm(nrow(dat))), ]

write.csv(dat, 'data/train_decode.csv', row.names=F, quote=F)

header = t(c('image',tmp))
write.table(header, 'data/submission_header.csv', 
            row.names=F, quote=F, col.names=F, sep=',')

summary(as.numeric(table(dat$label)))
print(length(labs))
