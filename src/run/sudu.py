#coding:utf-8#Python中声明文件编码的注释，编码格式指定为utf-8
from socket import *
from time import ctime
import binascii
import time
import threading

import RPi.GPIO as GPIO
class sudu(object):
    def __init__(self):
        IN1 = 19
        IN2 = 16
        IN3 = 21
        IN4 = 26
        ENA = 13
        ENB = 20
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(ENA,GPIO.OUT,initial=GPIO.HIGH)
        GPIO.setup(ENB,GPIO.OUT,initial=GPIO.HIGH)

        GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
        GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)
        GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
        GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
        GPIO.setup(IN1,GPIO.OUT)
        self.p1 = GPIO.PWM(IN1,200)
        GPIO.setup(IN3,GPIO.OUT)
        self.p3 = GPIO.PWM(IN3,200)
        GPIO.setup(IN2,GPIO.OUT)
        self.p2 = GPIO.PWM(IN2,200)
        GPIO.setup(IN4,GPIO.OUT)
        self.p4 = GPIO.PWM(IN4,200)
    

    def back(self,speed=100):
        print ("motor back")
        (self.p1).start(speed)
        (self.p3).start(speed)
        (self.p2).stop() # 停止PWM信号
        (self.p4).stop() # 停止PWM信号


    def gogo(self,speed=100):
        print ("motor_gogo")
        (self.p2).start(speed)
        (self.p4).start(speed)
        (self.p1).stop() # 停止PWM信号
        (self.p3).stop() # 停止PWM信号


    def right(self,speed=100):
        print ("motor_right")
        (self.p2).start(speed)
        (self.p3).start(speed)
        (self.p1).stop() # 停止PWM信号
        (self.p4).stop() # 停止PWM信号
    
    def left(self,speed=100):
        print ("motor_left")
        (self.p1).start(speed)
        (self.p4).start(speed)
        (self.p2).stop() # 停止PWM信号
        (self.p3).stop() # 停止PWM信号
    

    def stop(self):
        print ("motor_stop")
        (self.p1).stop() # 停止PWM信号
        (self.p3).stop() # 停止PWM信号
        (self.p2).stop() # 停止PWM信号
        (self.p4).stop() # 停止PWM信号
    

