// my.cpp
#include "test/test_ae2_trainer.h" 
#include "projects/python/graphics/graphics.h" 
//#include "test/test_anchor.h"
#include "test/test_autoencoder_new.h"
//#include "test/test_molecular.h"
//#include "test/test_iterator.h"
//#include "test/test_perf_molecular.h"
//#include "test/test_autoencoder3.h"
//#include "test/test_py_background.h"
//#include "test/test_autoencoder2.h"
//#include "test/test_mnist3.h"

//include "test/test_draw.h"
//#include "test/test_tensor.h"
//#include "test/test_image.h"
//#include "test/test_xml.h"
//#include "test/test_ai2.h"
//#include "test/test_numpy.h"
//#include "test/test_vuetify.h"
//#include "test/test_vue.h"
//#include "test/test_json.h"
//#include "test/parse_test.h"
//#include "test/std_test.h"
//#include "test/test_webview.h"
//#include "test/test_dat.h"
//#include "test\test_py.h"
//#include "test\lua_test.h"
//#include "test\molecularengine2.h"
//#include "test\engine2.h"
//#define USE_MY_CONTROLS
//#include "test/uitest.h"


myEngine* selectEngine(const char* app) {
    debug << "Please Select Function:\n";
    debug << "1 - Trainer\n";
    debug << "2 - Predictor\n";
    debug << "3 - Param Edit\n";
    int f = getchar();
    debug << f << "\n";
    if (f == '1')
		return new MyTrainer(app);

    if (f == '2')
	return new MyPredictor(app);	

    return new MyParamTest(app);
}


int main(int argc, char *argv[])
{   
    debug << "AutoEncoder  Trainer-Predictor\n";
    myEngine* engine = nullptr;
    if (argc < 2) {
        engine = selectEngine(argv[0]);
    }
    
    if (engine)
        engine->Start();

    delete engine;
    return 0;
}
