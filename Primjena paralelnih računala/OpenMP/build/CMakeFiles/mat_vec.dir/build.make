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
CMAKE_SOURCE_DIR = /.home/student1/imrovmir/PPR/OpenMP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /.home/student1/imrovmir/PPR/OpenMP/build

# Include any dependencies generated for this target.
include CMakeFiles/mat_vec.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mat_vec.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mat_vec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mat_vec.dir/flags.make

CMakeFiles/mat_vec.dir/mat_vec.cpp.o: CMakeFiles/mat_vec.dir/flags.make
CMakeFiles/mat_vec.dir/mat_vec.cpp.o: /.home/student1/imrovmir/PPR/OpenMP/mat_vec.cpp
CMakeFiles/mat_vec.dir/mat_vec.cpp.o: CMakeFiles/mat_vec.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/.home/student1/imrovmir/PPR/OpenMP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mat_vec.dir/mat_vec.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mat_vec.dir/mat_vec.cpp.o -MF CMakeFiles/mat_vec.dir/mat_vec.cpp.o.d -o CMakeFiles/mat_vec.dir/mat_vec.cpp.o -c /.home/student1/imrovmir/PPR/OpenMP/mat_vec.cpp

CMakeFiles/mat_vec.dir/mat_vec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mat_vec.dir/mat_vec.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /.home/student1/imrovmir/PPR/OpenMP/mat_vec.cpp > CMakeFiles/mat_vec.dir/mat_vec.cpp.i

CMakeFiles/mat_vec.dir/mat_vec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mat_vec.dir/mat_vec.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /.home/student1/imrovmir/PPR/OpenMP/mat_vec.cpp -o CMakeFiles/mat_vec.dir/mat_vec.cpp.s

# Object files for target mat_vec
mat_vec_OBJECTS = \
"CMakeFiles/mat_vec.dir/mat_vec.cpp.o"

# External object files for target mat_vec
mat_vec_EXTERNAL_OBJECTS =

mat_vec: CMakeFiles/mat_vec.dir/mat_vec.cpp.o
mat_vec: CMakeFiles/mat_vec.dir/build.make
mat_vec: CMakeFiles/mat_vec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/.home/student1/imrovmir/PPR/OpenMP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mat_vec"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mat_vec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mat_vec.dir/build: mat_vec
.PHONY : CMakeFiles/mat_vec.dir/build

CMakeFiles/mat_vec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mat_vec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mat_vec.dir/clean

CMakeFiles/mat_vec.dir/depend:
	cd /.home/student1/imrovmir/PPR/OpenMP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /.home/student1/imrovmir/PPR/OpenMP /.home/student1/imrovmir/PPR/OpenMP /.home/student1/imrovmir/PPR/OpenMP/build /.home/student1/imrovmir/PPR/OpenMP/build /.home/student1/imrovmir/PPR/OpenMP/build/CMakeFiles/mat_vec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mat_vec.dir/depend

