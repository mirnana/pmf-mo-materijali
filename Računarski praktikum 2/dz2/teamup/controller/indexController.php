<?php 
	session_start();

	class IndexController extends BaseController {
		public function index() {
			header('Location: ' . __SITE_URL . '/index.php?rt=login'); 
		}
	}; 

?>
