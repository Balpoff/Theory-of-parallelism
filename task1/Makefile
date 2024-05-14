CC=g++
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=my_executable

ifeq ($(DATA_TYPE),DOUBLE)
	CFLAGS+=-DDATA_TYPE_DOUBLE
else
ifeq ($(DATA_TYPE),FLOAT)
	CFLAGS+=-DDATA_TYPE_FLOAT
endif
endif

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
