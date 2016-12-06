TARGET 		= opengl

CC 			= gcc

CFLAGS 		= -W -Wall -O3 -fopenmp

LINKER 		= gcc -o

LFLAGS 		= -Wall -lGLESv2 -lEGL -lm -lX11 #-lEGL #-lSDL2# -fopenmp #-lGLEW -lGL -lGLU -lGLESv2

SRCDIR 		= src
OBJDIR 		= obj
BINDIR 		= bin

SOURCES 	:= $(wildcard $(SRCDIR)/*.c)
INCLUDES 	:= $(wildcard $(SRCDIR)/*.h)
OBJECTS 	:= $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
rm 			= rm -f

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS)
	@$(LINKER) $@ $(OBJECTS) $(LFLAGS) 

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.c
	@$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	@$(rm) $(OBJECTS)

.PHONY: remove
remove: clean
	@$(rm) $(BINDIR)/$(TARGET)

