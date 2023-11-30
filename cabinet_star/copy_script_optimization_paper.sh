#!/bin/sh

mkdir copy
cd copy

rm -rf source
rm -rf segmentation
mkdir source
mkdir segmentation


add_branch () {
    echo b-r-a-n-c-h $1
    echo 1 $1

    git clone --depth 1 ssh://git@navbitbucket01.navinfo.eu:7999/one/cabinet_star.git -b $1 source
    rm -rf source/.git source/.gitignore
    mkdir segmentation/$1
    mv source/* segmentation/$1/
}

add_branch master
add_branch AISEG-447-bisenet-one-branch
add_branch AISEG-398-Implement-shift-invariant-pytorch
add_branch AISEG-15-using-switchable-normalization-in-contextnet

rm -rf source
tar cvfz ../segmentation_code.tgz segmentation/*
cd ..

