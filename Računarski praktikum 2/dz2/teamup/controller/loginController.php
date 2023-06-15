<?php
    session_start();
    
    class LoginController extends BaseController {
        public function index() {
            $this->registry->template->title = 'Prijava u sustav';

            $this->registry->template->show('login_index');
        }

        public function login() {
            $tus = new TeamUpService();
            
            $user = $_POST['username'];
            $pass = $_POST['password'];

            $validUser = $tus->authentication($user, $pass);
            if($validUser.is_null()) {
                // loš login => vrati se na login_index
                $this->registry->template->title = 'Neispravni podaci, pokušaj ponovno';
                $this->registry->template->show('login_index');
            }
            else {console.log("here");
                // dobar login => prikaži popis projekata
                //session_start();
                $_SESSION['id_user'] = $validUser->id;
                $_SESSION['username'] = $validUser->username;
                $_SESSION['password'] = $validUser->password;
                $_SESSION['email'] = $validUser->email;
                $_SESSION['registration_sequence'] = $validUser->registration_sequence;
                $_SESSION['has_registered'] = $validUser->has_registered;
                $this->registry->template->show('projects_index');
            }
        }

        public function logout() {
            session_unset();
            session_destroy();
            $this->registry->template->show('login_index');
        }
    }
?>