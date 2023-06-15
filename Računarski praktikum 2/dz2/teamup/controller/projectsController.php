<?php
    session_start();
    
    class ProjectsController {
        function index() {
            $tus = new TeamUpService();

            $this->registry->template->title = 'Popis svih projekata';
            $this->registry->template->projectList = $tus->getAllProjects();

            $this->registry->template->show('projects_index');
        }

        function showProjectByTitle() {
            $tus = new TeamUpService();
            
            $project_id = $_POST['project_id']; 
            $project = $tus->getProjectByID($project_id);

            $this->registry->template->title = 'Detalji projekta';

            $this->registry->template->author_username              = $project['author_username'];
            $this->registry->template->project_title                = $project['project_title'];
            $this->registry->template->abstract                     = $project['abstract'];
            $this->registry->template->number_of_members            = $project['number_of_members'];
            $this->registry->template->current_number_of_members    = $project['current_number_of_members'];

            $this->registry->template->show('project_view');
        }

        function showMyProjects() {
            $tus = new TeamUpService();

            $user_id = $_SESSION['id_user'];

            $this->registry->template->title = 'Popis mojih projekata';
            $this->registry->template->projectList = $tus->getMyProjects($user_id);

            $this->registry->template->show('my_projects_index');
        }

        function applyForProject() {
            $tus = new TeamUpService();

            $project_id = $_POST['project_id'];
            $user_id = $_SESSION['id_user'];

            if(!isset($project_id) || !isset($user_id)){
                header('Location: ' . __SITE_URL . '/index.php?rt=projects_index');
                exit();
            }
            
            $project = $tus->getProjectByID($project_id);
            $available_slots = $project['number_of_members'] - $project['current_number_of_members'];

            if($available_slots > 0 && !($tus->memberOf($user_id, $project_id))) {
                $tus->applicationForProject($project_id, $user_id);

                if($available_slots === 1) $tus->closeProject($project_id);

                $this->registry->template->title = 'Uspješna prijava';
                $this->registry->template->sentence = 'Prijavljeni ste na projekt ' . $project['project_title'] . '.';
            }
            else {
                $this->registry->template->title = 'Neuspješna prijava';
                $this->registry->template->sentence = 'Prijava na projekt ' . $project['project_title'] . ' nije uspjela.';
            }

            $this->registry->template->show('project_application');
            // u view dodati link na stranicu projekta, nakon što se prikažu stranice (Ne)uspješna prijava
        }

        function newProjectForm() {
            $this->registry->template->title = 'Unos novog projekta';
            $this->registry->template->show('project_form');
        }

        function createProject() {
            $tus = new TeamUpService();

            $user_id          = $_SESSION['id_user'];
            $project_title    = $_POST['project_title'];
            $project_abstract = $_POST['project_abstract'];
            $project_number   = $_POST['project_number'];

            $project_status = 'open';
            if($project_number <= 1) $project_status = 'closed';
            $tus->enterNewProject($user_id
                                , $project_title
                                , $project_abstract
                                , $project_number
                                , $project_status);
            
            $this->registry->template->show('projects_index');
        }
    }
?>