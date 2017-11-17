##### Paths ####
pwd = $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
obj = $(pwd:%raja=%obj)
okina = $(pwd:%raja=%)

#### RAJA ####
raja = $(pwd)

#### RAJA source files & objects ####
crF = $(wildcard $(raja)/*.cpp)
orF = $(patsubst $(raja)%.cpp,$(obj)%.ro,$(crF))

#### Object files ####
objects = $(orF) # RAJA

#### MFEM & OCCA paths ####
MFEM = /home/camier1/home/mfem/mfem-okina

#### Compilation flags ####
CXX = g++
CXX = mpicxx
CXXFLAGS = -fPIC -rdynamic -Wall -g
INCLUDES = -I$(okina)/raja -I$(okina)/kernels -I$(MFEM)/fem -I/home/camier1/home/dbg
CPU = $(shell echo $(shell getconf _NPROCESSORS_ONLN)*2|bc -l)

#### debug variables ####
COLOR_OFFSET = 16
COLOR = $(shell echo $(rule_path)|cksum|cut -b1-2)
rule_path = $(notdir $(patsubst %/,%,$(dir $<)))
rule_file = $(basename $(notdir $@))
rule_dumb = @echo -e make: $(rule_path)/$(rule_file)
rule_xterm = @echo -e \\e[38\;5\;$(shell echo $(COLOR)+$(COLOR_OFFSET)|bc -l)\;1m\
             $(rule_path)\\033[m/\\033[\m$(rule_file)\\033[m
rule = $(rule_${TERM})

#### quiet ####
quiet := --quiet -S

#### PHONIES ####
.PHONY = clean cln all

####
all: | $(shell mkdir -p $(obj))#;$(rule)
	@$(MAKE) $(quiet) -j $(CPU) $(objects)
	@cd .. && $(MAKE) $(quiet) # make libokina.so

#### RAJA Sources #####
$(obj)/%.ro: $(raja)/%.cpp $(raja)/%.hpp $(raja)/*.hpp;$(rule)
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<

#### Clean ####
clean cln:;$(rule)
	rm -rf $(objects)
