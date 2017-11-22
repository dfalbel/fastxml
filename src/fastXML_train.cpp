#include <iostream>
#include <fstream>
#include <string>

#include "fastXML.h"
#include "timer.h"
using namespace std;

Param parse_param(_int argc, char* argv[])
{
	Param param;

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);

		if(opt=="-m")
			param.max_leaf = (_int)val;
		else if(opt=="-l")
			param.lbl_per_leaf = (_int)val;
		else if(opt=="-b")
			param.bias = (_float)val;
		else if(opt=="-c")
			param.log_loss_coeff = (_float)val;
		else if(opt=="-T")
			param.num_thread = (_int)val;
		else if(opt=="-s")
			param.start_tree = (_int)val;
		else if(opt=="-t")
			param.num_tree = (_int)val;
		else if(opt=="-q")
			param.quiet = (_bool)val;
	}

	return param;
}

int main(int argc, char* argv[])
{
	string ft_file = string(argv[1]);
	check_valid_filename(ft_file, true);
	SMatF* trn_ft_mat = new SMatF(ft_file);

	string lbl_file = string(argv[2]);
	check_valid_filename(lbl_file, true);
	SMatF* trn_lbl_mat = new SMatF(lbl_file);

	string model_folder = string(argv[3]);
	check_valid_foldername(model_folder);

	Param param = parse_param(argc-4,argv+4);
	param.num_ft = trn_ft_mat->nr;
	param.num_lbl = trn_lbl_mat->nr;
	param.write(model_folder+"/param");

	if( param.quiet )
		loglvl = LOGLVL::QUIET;

	Timer timer;
	timer.start();
	train_trees(trn_ft_mat, trn_lbl_mat, param, model_folder);
	cout << "training time: " << timer.stop() << " s" << endl;

	delete trn_ft_mat;
	delete trn_lbl_mat;
}
