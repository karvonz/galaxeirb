# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#   (C) COPYRIGHT 2013 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.

ROOT:=../..

CC:=g++


	RM:=rm -f
	CONCATENATE:=;
	MKDIR:=mkdir -p
	CP:=cp
CFLAGS:=-c -Wall -I$(ROOT)/include -I. -O3

LDFLAGS:=-L$(ROOT)/lib  -lOpenCL  -lGLESv2 -lEGL -lm -lOpenCL -lX11

SRCDIR 		= src
OBJDIR 		= obj
BINDIR 		= bin

SOURCES:=$(wildcard $(SRCDIR)/*.cpp)
HEADERS:=$(wildcard $(SRCDIR)/*.h)

OBJECTS:=$(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

EXECUTABLE:=galaxeirb

all: $(BINDIR)/$(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) libOpenCL 
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@$(CC) $(CFLAGS) -c $< -o $@

install: $(EXECUTABLE)
	-$(MKDIR) "$(ROOT)/bin/$(EXECUTABLE)/assets"
	$(CP) "$(EXECUTABLE)" "$(ROOT)/bin/$(EXECUTABLE)/$(EXECUTABLE)"
	cd assets $(CONCATENATE) $(CP) * "../$(ROOT)/bin/$(EXECUTABLE)/assets/"

.PHONY: clean libOpenCL

clean:
	$(RM) $(OBJECTS) $(EXECUTABLE)

libOpenCL:
	cd $(ROOT)/lib $(CONCATENATE) $(MAKE) libOpenCL.so
