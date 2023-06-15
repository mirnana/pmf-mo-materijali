<?php 
    class Projects {
        
        protected $id
                , $id_user
                , $title
                , $abstract
                , $number_of_members
                , $status;
        
        function __construct ($id, $id_user, $title, $abstract, $number_of_members, $status) {
            $this->id                = $id;
            $this->id_user           = $id_user;
            $this->title             = $title;
            $this->abstract          = $abstract;
            $this->number_of_members = $number_of_members;
            $this->status            = $status;
        }

        function __get( $prop ) { 
            return $this->$prop; 
        }
	    function __set( $prop, $val ) { 
            $this->$prop = $val; return $this; 
        }
    }
?>