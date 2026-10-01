// ==========================================
// Credit Card Risk Prediction
// Main JavaScript
// ==========================================


document.addEventListener(
    "DOMContentLoaded",
    function () {

        console.log(
            "Credit Card Risk Prediction application loaded."
        );


        // ==================================
        // BUTTON HOVER EFFECT
        // ==================================

        const buttons =
            document.querySelectorAll(
                ".predict-button"
            );


        buttons.forEach(
            function (button) {

                button.addEventListener(
                    "click",
                    function () {

                        console.log(
                            "Prediction process started."
                        );

                    }
                );

            }
        );


        // ==================================
        // FORM VALIDATION
        // ==================================

        const form =
            document.querySelector("form");


        if (form) {

            form.addEventListener(
                "submit",
                function (event) {

                    const age =
                        document.getElementById("age");


                    const employment =
                        document.getElementById(
                            "employment_years"
                        );


                    // Check age

                    if (age) {

                        const ageValue =
                            Number(age.value);


                        if (
                            ageValue < 18 ||
                            ageValue > 100
                        ) {

                            alert(
                                "Please enter an age between 18 and 100."
                            );

                            event.preventDefault();

                            return;

                        }

                    }


                    // Check employment

                    if (employment) {

                        const employmentValue =
                            Number(
                                employment.value
                            );


                        if (
                            employmentValue < 0 ||
                            employmentValue > 60
                        ) {

                            alert(
                                "Please enter employment experience between 0 and 60 years."
                            );

                            event.preventDefault();

                            return;

                        }

                    }

                }
            );

        }

    }
);