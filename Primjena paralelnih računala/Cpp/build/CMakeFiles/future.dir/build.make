# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/linuxbrew/.linuxbrew/Cellar/cmake/3.24.3/bin/cmake

# The command to remove a file.
RM = /home/linuxbrew/.linuxbrew/Cellar/cmake/3.24.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /.home/student1/imrovmir/PPR/Cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /.home/student1/imrovmir/PPR/Cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/future.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/future.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/future.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/future.dir/flags.make

CMakeFiles/future.dir/future.cpp.o: CMakeFiles/future.dir/flags.make
CMakeFiles/future.dir/future.cpp.o: /.home/student1/imrovmir/PPR/Cpp/future.cpp
CMakeFiles/future.dir/future.cpp.o: CMakeFiles/future.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/.home/student1/imrovmir/PPR/Cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/future.dir/future.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/future.dir/future.cpp.o -MF CMakeFiles/future.dir/future.cpp.o.d -o CMakeFiles/future.dir/future.cpp.o -c /.home/student1/imrovmir/PPR/Cpp/future.cpp

CMakeFiles/future.dir/future.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/future.dir/future.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /.home/student1/imrovmir/PPR/Cpp/future.cpp > CMakeFiles/future.dir/future.cpp.i

CMakeFiles/future.dir/future.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/future.dir/future.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /.home/student1/imrovmir/PPR/Cpp/future.cpp -o CMakeFiles/future.dir/future.cpp.s

# Object files for target future
future_OBJECTS = \
"CMakeFiles/future.dir/future.cpp.o"

# External object files for target future
future_EXTERNAL_OBJECTS =

future: CMakeFiles/future.dir/future.cpp.o
future: CMakeFiles/future.dir/build.make
future: CMakeFiles/future.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/.home/student1/imrovmir/PPR/Cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable future"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/future.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/future.dir/build: future
.PHONY : CMakeFiles/future.dir/build

CMakeFiles/future.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/future.dir/cmake_clean.cmake
.PHONY : CMakeFiles/future.dir/clean

CMakeFiles/future.dir/depend:
	cd /.home/student1/imrovmir/PPR/Cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /.home/student1/imrovmir/PPR/Cpp /.home/student1/imrovmir/PPR/Cpp /.home/student1/imrovmir/PPR/Cpp/build /.home/student1/imrovmir/PPR/Cpp/build /.home/student1/imrovmir/PPR/Cpp/build/CMakeFiles/future.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/future.dir/depend

